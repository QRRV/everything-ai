# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-initialization: init_std, the spread of the starting weights by He or by Xavier, and
he_init, which gives a list of PyTorch layers He starting weights and bias terms of 0. And the picture: the spread of
the activations through 20 ReLU layers, with He and with Xavier starting weights."""
import math
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai.check import ClassBox, FunctionBox, Mistake


def spreads(kind: str, layers: int = 20, width: int = 256, seed: int = 0) -> list[float]:
    """The standard deviation of the activations after each of `layers` ReLU layers, for a batch of 512 random
    inputs, with He or Xavier starting weights and bias terms of 0."""
    torch.manual_seed(seed)
    stack = [nn.Linear(width, width) for _ in range(layers)]
    for layer in stack:
        if kind == "he":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
    h = torch.randn(512, width)
    out = []
    with torch.no_grad():
        for layer in stack:
            h = torch.relu(layer(h))
            out.append(h.std().item())
    return out


def spread_picture(labels: dict) -> str:
    from everything_ai import chart
    steps = list(range(1, 21))
    panel = chart.Panel((0.5, 20.5), (1e-4, 10), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylog=True,
                        xticks=[1, 5, 10, 15, 20])
    panel.line(steps, spreads("he"), "hidden", 2.4)
    panel.line(steps, spreads("xavier"), "hidden", 2.0, dashed=True)
    panel.text(13, 2.0, labels["he"], "hidden", "middle")
    panel.text(9, 0.004, labels["xavier"], "hidden", "middle")
    return chart.figure([panel], alt=labels["alt"], width=560, height=300)


# The lesson is done when these are correct. "he-512" is the choice question in the lesson.
REQUIRED = ["he-512", "init-std", "he-init"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _num(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6)
    except (TypeError, ValueError):
        return False


# --- exercise 2: the spread -------------------------------------------------------------------------------------------------

_S_HEAD = "def init_std(n_in, n_out, activation):\n"
_S_START = _S_HEAD + '    # the standard deviation of the starting weights: He for "relu", Xavier for "tanh"\n    ...\n'


def _s(he: str = "(2 / n_in) ** 0.5", xavier: str = "(2 / (n_in + n_out)) ** 0.5") -> str:
    return _S_HEAD + f'    if activation == "relu":\n        return {he}\n    return {xavier}\n'


# --- exercise 3: He starting weights for a list of layers ------------------------------------------------------------------

_H_HEAD = "import torch\nfrom torch import nn\n\n\ndef he_init(layers):\n"
_H_START = _H_HEAD + "    # every layer: He starting weights, and bias terms of 0\n    ...\n"


def _h(weight: str = 'nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")', bias: str = "nn.init.zeros_(layer.bias)") -> str:
    return _H_HEAD + f"    for layer in layers:\n        {weight}\n        {bias}\n"


_LAYERS = "torch.manual_seed(0)\nlayers = [nn.Linear(400, 100), nn.Linear(100, 50)]\nhe_init(layers)\n"
# A loop, not a list comprehension: py:comprehension is not in the requires chain of this lesson.
_STDS = "stds = []\nfor layer in layers:\n    stds.append(round(layer.weight.std().item(), 3))\nstds"
_BIASES = "largest = []\nfor layer in layers:\n    largest.append(round(layer.bias.abs().max().item(), 3))\nlargest"
# The smallest layer has 5,000 weights: the spread of another random draw stays well within 6%.
_H_SCENARIOS = [
    _LAYERS + _STDS,
    _LAYERS + _BIASES,
    "torch.manual_seed(1)\nlayers = [nn.Linear(50, 200), nn.Linear(200, 200), nn.Linear(200, 100)]\nhe_init(layers)\n" + _STDS,
]


def _spread_alike(got, want) -> bool:
    """Spreads of random weights are alike within 6%: another way of drawing the same spread gives other numbers."""
    try:
        return len(got) == len(want) and all(abs(float(g) - float(w)) <= 0.06 * max(abs(float(w)), 1e-9) or float(g) == float(w)
                                              for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "init-std": FunctionBox(
        name="init-std",
        function="init_std",
        start=_S_START,
        solution=_s(),
        visible=[(200, 100, "relu")],
        hidden=[(200, 100, "tanh"), (784, 256, "relu"), (64, 64, "tanh"), (10, 30, "tanh")],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("he-no-two", wrong=_s(he="(1 / n_in) ** 0.5"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the spread that keeps the signal for an activation function that lets all of it through. ReLU sets about half to 0, so He doubles the variance: the square root of `2 / n_in`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de spreiding die het signaal houdt bij een activation function die alles doorlaat. ReLU zet ongeveer de helft op 0, dus He verdubbelt de variantie: de wortel uit `2 / n_in`."),
            Mistake("variance", wrong=_s(he="2 / n_in", xavier="2 / (n_in + n_out)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the variance. The standard deviation is its square root: `** 0.5`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de variantie. De standaardafwijking is de wortel daaruit: `** 0.5`."),
            Mistake("swapped", wrong=_s(he="(2 / (n_in + n_out)) ** 0.5", xavier="(2 / n_in) ** 0.5"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: He and Xavier are the other way round. He is for ReLU, and counts only the inputs; Xavier is for tanh, and counts inputs and outputs.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: He en Xavier staan omgekeerd. He is voor ReLU, en telt alleen de inputs; Xavier is voor tanh, en telt inputs en outputs."),
            Mistake("xavier-in", wrong=_s(xavier="(1 / n_in) ** 0.5"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: Xavier counts the inputs and the outputs, so that the signal keeps its size on the way forward and the gradient on the way back: the square root of `2 / (n_in + n_out)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: Xavier telt de inputs en de outputs, zodat het signaal op de heenweg even groot blijft en de gradient op de terugweg: de wortel uit `2 / (n_in + n_out)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: one spread for \"relu\", another for \"tanh\".",
                      "nl": "Vervang de `...` door de body van de functie: één spreiding voor \"relu\", een andere voor \"tanh\"."},
            "arguments": {"en": "The check calls `{call}`, with the inputs, the outputs and the activation, and the `def` line of your function has another number of names between its parentheses. Keep `(n_in, n_out, activation)`.",
                          "nl": "De check roept `{call}` aan, met de inputs, de outputs en de activation function, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n_in, n_out, activation)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Which numbers does the variance count for this activation, and is your answer its square root?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Welke aantallen telt de variantie bij deze activation function, en is je antwoord de wortel daaruit?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function also give the spread for \"tanh\"?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Geeft je functie ook de spreiding voor \"tanh\"?"},
            "correct": {"en": "Your function is right in all {n} cases: He for ReLU, Xavier for tanh.",
                        "nl": "Je functie klopt in alle {n} gevallen: He voor ReLU, Xavier voor tanh."},
        },
    ),
    "he-init": ClassBox(
        name="he-init",
        start=_H_START,
        solution=_h(),
        scenarios=_H_SCENARIOS,
        compare=_spread_alike,
        first=[_DOTS],
        mistakes=[
            Mistake("no-two", wrong=_h(weight="nn.init.normal_(layer.weight, 0, (1 / layer.in_features) ** 0.5)"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nthe spreads are `{got}`, and they should be about `{want}`: about √2 times too small. He keeps the 2 for ReLU: the square root of `2 / n_in`, as `nn.init.kaiming_normal_` does.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nzijn de spreidingen `{got}`, en dat moet ongeveer `{want}` zijn: ongeveer √2 keer te klein. He houdt de 2 voor ReLU: de wortel uit `2 / n_in`, zoals `nn.init.kaiming_normal_` doet."),
            Mistake("fan-out", wrong=_h(weight='nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")'),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nthe spreads are `{got}`, and they should be about `{want}`: you count the outputs of each layer. He counts its inputs, `layer.in_features`, the numbers each neuron adds up.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nzijn de spreidingen `{got}`, en dat moet ongeveer `{want}` zijn: je telt de outputs van elke layer. He telt zijn inputs, `layer.in_features`, de getallen die elk neuron optelt."),
            Mistake("xavier", wrong=_h(weight="nn.init.xavier_normal_(layer.weight)"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nthe spreads are `{got}`, and they should be about `{want}`: that is Xavier, made for tanh. For ReLU, use He: `nn.init.kaiming_normal_(layer.weight, nonlinearity=\"relu\")`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nzijn de spreidingen `{got}`, en dat moet ongeveer `{want}` zijn: dat is Xavier, gemaakt voor tanh. Gebruik voor ReLU He: `nn.init.kaiming_normal_(layer.weight, nonlinearity=\"relu\")`."),
            Mistake("he-variance", wrong=_h(weight="nn.init.normal_(layer.weight, 0, 2 / layer.in_features)"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nthe spreads are `{got}`, and they should be about `{want}`: you give `normal_` the variance, `2 / n_in`. It takes the standard deviation, the square root of that: `(2 / layer.in_features) ** 0.5`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nzijn de spreidingen `{got}`, en dat moet ongeveer `{want}` zijn: je geeft `normal_` de variantie, `2 / n_in`. Het neemt de standaardafwijking, de wortel daaruit: `(2 / layer.in_features) ** 0.5`."),
            Mistake("bias-kept", wrong=_h(bias="pass"), state="almost",
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nthe largest bias terms are `{got}`, and they should be `{want}`: the bias terms still have the random starting values of `nn.Linear`. Set them to 0: `nn.init.zeros_(layer.bias)`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nzijn de grootste bias terms `{got}`, en dat moet `{want}` zijn: de bias terms hebben nog de willekeurige beginwaarden van `nn.Linear`. Zet ze op 0: `nn.init.zeros_(layer.bias)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with a loop over the layers: for each, He starting weights and bias terms of 0.",
                      "nl": "Vervang de `...` door een loop over de layers: voor elke layer He-beginweights en bias terms van 0."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be about `{want}`. He gives the weights of each layer a spread of the square root of `2 / n_in`, with `n_in` its number of inputs.",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet ongeveer `{want}` zijn. He geeft de weights van elke layer als spreiding de wortel uit `2 / n_in`, met `n_in` zijn aantal inputs."},
            "almost": {"en": "Right for the spreads of the example. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be about `{want}`. Are all bias terms 0, and does every layer get the spread of its own number of inputs?",
                       "nl": "Goed voor de spreidingen van het voorbeeld. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet ongeveer `{want}` zijn. Zijn alle bias terms 0, en krijgt elke layer de spreiding van zijn eigen aantal inputs?"},
            "correct": {"en": "Every layer has He starting weights, with the spread of its own number of inputs, and bias terms of 0.",
                        "nl": "Elke layer heeft He-beginweights, met de spreiding van zijn eigen aantal inputs, en bias terms van 0."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("init-std", _S_START, "empty", None),
    ("init-std", _s(), "correct", None),
    ("init-std", "import math\n\n\n" + _S_HEAD + '    if activation == "relu":\n        return math.sqrt(2 / n_in)\n    else:\n        return math.sqrt(2 / (n_in + n_out))\n', "correct", None),
    ("init-std", _s(he="(1 / n_in) ** 0.5"), "wrong", "he-no-two"),
    ("init-std", _s(he="2 / n_in", xavier="2 / (n_in + n_out)"), "wrong", "variance"),
    ("init-std", _s(he="(2 / (n_in + n_out)) ** 0.5", xavier="(2 / n_in) ** 0.5"), "wrong", "swapped"),
    ("init-std", _s(xavier="(1 / n_in) ** 0.5"), "almost", "xavier-in"),
    ("he-init", _H_START, "empty", None),
    ("he-init", _h(), "correct", None),
    ("he-init", _h(weight="nn.init.normal_(layer.weight, 0, (2 / layer.in_features) ** 0.5)", bias="nn.init.zeros_(layer.bias)"), "correct", None),
    ("he-init", _H_HEAD + "    with torch.no_grad():\n        for layer in layers:\n            layer.weight.copy_(torch.randn(layer.out_features, layer.in_features) * (2 / layer.in_features) ** 0.5)\n            layer.bias.zero_()\n", "correct", None),
    ("he-init", _h(weight="nn.init.normal_(layer.weight, 0, 2 / layer.in_features)"), "wrong", "he-variance"),
    ("he-init", _h(weight="nn.init.normal_(layer.weight, 0, (1 / layer.in_features) ** 0.5)"), "wrong", "no-two"),
    ("he-init", _h(weight='nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")'), "wrong", "fan-out"),
    ("he-init", _h(weight="nn.init.xavier_normal_(layer.weight)"), "wrong", "xavier"),
    ("he-init", _h(bias="pass"), "almost", "bias-kept"),
    ("he-init", _h(bias="nn.init.ones_(layer.bias)"), "almost", None),
]
