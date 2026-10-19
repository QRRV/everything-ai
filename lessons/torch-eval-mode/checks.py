# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-eval-mode: a function `evaluate` that switches a network to eval mode, works out
the loss without recording a graph, and switches it back; and which lines belong to training, to predicting, or to
both.
And the picture: a small network with dropout, its predictions in train mode, three times, and in eval mode."""
import functools
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai.check import CodeBox, Mistake, Sort

# --- the data, the network and the picture -------------------------------------------------------------------------


def data():
    """200 examples with x between -2 and 2 and y = 2x + 1 plus random scatter, as in torch-batches."""
    torch.manual_seed(0)
    X = torch.rand(200, 1) * 4 - 2
    y = 2 * X + 1 + 0.5 * torch.randn(200, 1)
    return X, y


class Net(nn.Module):
    """1 input, a hidden layer of 16 neurons with ReLU, dropout with p = 0.5, and 1 output."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 16)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        return self.out(self.drop(torch.relu(self.hidden(x))))


@functools.lru_cache(maxsize=1)
def trained():
    """The network after 15 epochs of mini-batches of 20, and its predictions on a grid: three times in train mode,
    once in eval mode, and the loss over all data in each mode. Worked out once."""
    X, y = data()
    torch.manual_seed(1)
    net = Net()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    loader = DataLoader(TensorDataset(X, y), batch_size=20, shuffle=True)
    for _ in range(15):
        for xb, yb in loader:
            loss = ((net(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    grid = torch.linspace(-2, 2, 41).reshape(41, 1)
    with torch.no_grad():
        train_curves = [net(grid).flatten().tolist() for _ in range(3)]
        train_loss = sum(((net(X) - y) ** 2).mean().item() for _ in range(50)) / 50   # a new draw at every call: the mean
        net.eval()
        eval_curve = net(grid).flatten().tolist()
        eval_loss = ((net(X) - y) ** 2).mean().item()
        net.train()
    return grid.flatten().tolist(), train_curves, eval_curve, train_loss, eval_loss


def picture(labels: dict[str, str]) -> str:
    """The 200 examples, three predictions of the network in train mode (thin) and one in eval mode (thick)."""
    from everything_ai import chart
    X, y = data()
    grid, train_curves, eval_curve, _, _ = trained()
    p = chart.Panel((-2, 2), (-4, 6), title=labels["title"], xlabel="x", ylabel="y")
    p.points(X.flatten().tolist(), y.flatten().tolist(), "data", 2.5)
    for curve in train_curves:
        p.line(grid, curve, "pred", 1.2, dashed=True)
    p.line(grid, eval_curve, "pred", 3)
    p.text(-1.95, 5.3, labels["dashed"], "ink", "start")
    p.text(-1.95, 4.4, labels["solid"], "ink", "start")
    return chart.figure([p], alt=labels["alt"], width=520, height=320)


# The lesson is done when these are correct. "two-switches" is the choice question in the lesson.
REQUIRED = ["two-switches", "evaluate", "when"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: evaluate -----------------------------------------------------------------------------------------------

_P_HEAD = ("import torch\nfrom torch import nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
           "        self.hidden = nn.Linear(1, 16)\n        self.drop = nn.Dropout(0.5)\n        self.out = nn.Linear(16, 1)\n\n"
           "    def forward(self, x):\n        return self.out(self.drop(torch.relu(self.hidden(x))))\n\n\n"
           "def evaluate(model, X, y):\n")
_P_END = ("\n\ntorch.manual_seed(1)\nnet = Net()\nX = torch.tensor([[0.0], [1.0], [2.0]])\ny = torch.tensor([[1.0], [3.0], [5.0]])\n"
          "loss = evaluate(net, X, y)\n[loss.requires_grad, net.training, loss.item()]")
_P_START = _P_HEAD + "    ...\n" + _P_END + "\n"


def _p(body: str = "    model.eval()\n    with torch.no_grad():\n        loss = ((model(X) - y) ** 2).mean()\n    model.train()\n    return loss\n") -> str:
    return _P_HEAD + body + _P_END


_P = _p()

EXERCISES = {
    "evaluate": CodeBox(
        name="evaluate",
        start=_P_START,
        solution=_P,
        keep=["def evaluate(model, X, y):", "torch.manual_seed(1)", "loss = evaluate(net, X, y)", "[loss.requires_grad, net.training, loss.item()]"],
        first=[_DOTS],
        mistakes=[
            Mistake("no-eval", wrong=_p("    with torch.no_grad():\n        loss = ((model(X) - y) ** 2).mean()\n    return loss\n"),
                    en="Your last line gives `{got}`: that is the loss of the network in train mode, where dropout sets numbers of the hidden layer to 0 at random. Switch to eval mode first, with `model.eval()`.",
                    nl="Je laatste regel geeft `{got}`: dat is de loss van het netwerk in train mode, waarin dropout willekeurig getallen van de hidden layer op 0 zet. Zet het netwerk eerst in eval mode, met `model.eval()`."),
            Mistake("recorded", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and got[0] is True,
                    en="Your last line gives `{got}`: `loss.requires_grad` is `True`, so PyTorch recorded a graph for the loss, which nobody needs: nothing is learned from it. Work out the loss inside `with torch.no_grad():`.",
                    nl="Je laatste regel geeft `{got}`: `loss.requires_grad` is `True`, dus PyTorch nam een graaf op voor de loss, en die heeft niemand nodig: er wordt niets van geleerd. Reken de loss uit binnen `with torch.no_grad():`."),
            Mistake("no-square", wrong=_p("    model.eval()\n    with torch.no_grad():\n        loss = (model(X) - y).mean()\n    model.train()\n    return loss\n"),
                    en="Your last line gives `{got}`: no graph and back in train mode, but the loss is the mean of the differences, not of the squared differences, so plus and minus cancel each other out. Square each difference first, with `** 2`.",
                    nl="Je laatste regel geeft `{got}`: geen graaf en weer in train mode, maar de loss is het gemiddelde van de verschillen, niet van de gekwadrateerde verschillen, dus plus en min vallen tegen elkaar weg. Kwadrateer eerst elk verschil, met `** 2`."),
            Mistake("item", error=lambda e: isinstance(e, AttributeError) and "'float' object has no attribute 'requires_grad'" in str(e),
                    en="Your `evaluate` gives back a plain number, not the tensor `loss`, and a plain number has no `requires_grad`. Give back `loss` itself, without `.item()`: the last line takes `.item()` itself.",
                    nl="Je `evaluate` geeft een gewoon getal terug, niet de tensor `loss`, en een gewoon getal heeft geen `requires_grad`. Geef `loss` zelf terug, zonder `.item()`: de laatste regel neemt zelf `.item()`."),
            Mistake("stays-eval", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and got[1] is False,
                    en="Your last line gives `{got}`: `net.training` is `False`, so the network stays in eval mode, and the next training step would run without dropout. Switch it back with `model.train()` before `return`: a line after `return` never runs.",
                    nl="Je laatste regel geeft `{got}`: `net.training` is `False`, dus het netwerk blijft in eval mode, en de volgende trainingsstap zou zonder dropout lopen. Zet het terug met `model.train()` vóór `return`: een regel na `return` wordt nooit uitgevoerd."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with steps 1 to 3 of the recipe, with the loss in step 2, and give back the loss.",
                      "nl": "Vervang de `...` door stap 1 tot en met 3 van het recept, met de loss in stap 2, en geef de loss terug."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. It should show no recorded graph, the network back in train mode, and the mean squared error of the network in eval mode. Do you follow steps 1 to 3 of the recipe, in that order?",
                      "nl": "Je laatste regel geeft `{got}`. Hij hoort geen opgenomen graaf te laten zien, het netwerk weer in train mode, en de mean squared error van het netwerk in eval mode. Volg je stap 1 tot en met 3 van het recept, in die volgorde?"},
            "correct": {"en": "No graph recorded, the network back in train mode, and the loss of the network as it predicts: the same at every call of `evaluate`, as dropout is off. So you can measure a network in between training steps without changing it.",
                        "nl": "Geen graaf opgenomen, het netwerk weer in train mode, en de loss van het netwerk zoals het voorspelt: bij elke aanroep van `evaluate` dezelfde, want dropout staat uit. Zo meet je een netwerk tussen trainingsstappen door zonder het te veranderen."},
        },
    ),
    "when": Sort(
        name="when",
        groups={
            "train": {"en": "Only for training", "nl": "Alleen voor het trainen"},
            "predict": {"en": "Only for predicting", "nl": "Alleen voor het voorspellen"},
            "both": {"en": "For both", "nl": "Voor allebei"},
        },
        items={
            "train-mode": {"en": "`model.train()`", "nl": "`model.train()`"},
            "eval-mode": {"en": "`model.eval()`", "nl": "`model.eval()`"},
            "zero": {"en": "`opt.zero_grad()`", "nl": "`opt.zero_grad()`"},
            "backward": {"en": "`loss.backward()`", "nl": "`loss.backward()`"},
            "no-grad": {"en": "`with torch.no_grad():` around the forward pass", "nl": "`with torch.no_grad():` om de forward pass heen"},
            "forward": {"en": "`y_pred = model(X)`", "nl": "`y_pred = model(X)`"},
        },
        solution={"train-mode": "train", "eval-mode": "predict", "zero": "train", "backward": "train", "no-grad": "predict",
                  "forward": "both"},
        nudge={
            "train-mode": {"en": "Which mode do layers such as dropout need while the network learns?",
                           "nl": "Welke mode hebben layers zoals dropout nodig terwijl het netwerk leert?"},
            "eval-mode": {"en": "When should dropout be off?",
                          "nl": "Wanneer moet dropout uit staan?"},
            "zero": {"en": "Are there slopes to clear when nothing is learned?",
                     "nl": "Zijn er hellingen te wissen als er niets geleerd wordt?"},
            "backward": {"en": "Does predicting need slopes?",
                         "nl": "Heeft voorspellen hellingen nodig?"},
            "no-grad": {"en": "Could a training step get its slopes if PyTorch recorded nothing of the forward pass?",
                        "nl": "Zou een trainingsstap zijn hellingen krijgen als PyTorch niets van de forward pass opnam?"},
            "forward": {"en": "Does a training step need the outputs of the network? And does a prediction?",
                        "nl": "Heeft een trainingsstap de outputs van het netwerk nodig? En een voorspelling?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the lines again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de regels opnieuw uit."},
            "empty": {"en": "Put each line in a group: only for training, only for predicting, or for both.",
                      "nl": "Zet elke regel in een groep: alleen voor het trainen, alleen voor het voorspellen, of voor allebei."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another group.",
                      "nl": "{item} hoort in een andere groep."},
            "correct": {"en": "All {n} are right. Training needs train mode, a recorded graph and the slopes; predicting needs eval mode and no graph; both need the forward pass.",
                        "nl": "Alle {n} kloppen. Trainen heeft train mode, een opgenomen graaf en de hellingen nodig; voorspellen eval mode en geen graaf; allebei de forward pass."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("evaluate", _P_START, "empty", None),
    ("evaluate", _p("    model.eval()\n    with torch.no_grad():\n        pred = model(X)\n        result = torch.mean((pred - y) ** 2)\n    model.train()\n    return result\n"), "correct", None),
    ("evaluate", _p("    with torch.no_grad():\n        model.eval()\n        loss = ((model(X) - y) ** 2).mean()\n        model.train()\n    return loss\n"), "correct", None),
    ("evaluate", _p("    with torch.no_grad():\n        loss = ((model(X) - y) ** 2).mean()\n    return loss\n"), "wrong", "no-eval"),
    ("evaluate", _p("    model.eval()\n    loss = ((model(X) - y) ** 2).mean()\n    model.train()\n    return loss\n"), "wrong", "recorded"),
    ("evaluate", _p("    model.eval()\n    with torch.no_grad():\n        loss = ((model(X) - y) ** 2).mean()\n    return loss\n"), "wrong", "stays-eval"),
    ("evaluate", _p("    model.eval()\n    with torch.no_grad():\n        return ((model(X) - y) ** 2).mean()\n    model.train()\n"), "wrong", "stays-eval"),
    ("evaluate", _p("    model.eval()\n    with torch.no_grad():\n        loss = (model(X) - y).mean()\n    model.train()\n    return loss\n"), "wrong", "no-square"),
    ("evaluate", _p("    model.eval()\n    with torch.no_grad():\n        loss = ((model(X) - y) ** 2).mean()\n    model.train()\n    return loss.item()\n"), "error", "item"),
]
