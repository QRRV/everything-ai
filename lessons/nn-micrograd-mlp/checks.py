# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-micrograd-mlp: the learner's own Module and Neuron, saved as nn/neuron.py, and Layer and
MLP on top of them, saved as nn/mlp.py, built on the learner's Value and backward, or on the course's where theirs is
missing or broken. And the demo: an untrained MLP(2, [4, 4, 1]) on the corners of XOR, and one backward pass through it."""
import re

from everything_ai import projects
from everything_ai.check import ClassBox, Mistake, run_code

# The Value and backward of the lessons before: yours when they are there and give the course's results, else the
# course's (template 1.3).
projects.builds_on(("nn", "value", "nn-micrograd-value", "value"), ("nn", "backward", "nn-micrograd-backward", "backward"))
_VALUE = projects.course_code("nn", "value")


def _same(module_lines: str, course: str, scenarios: list[str]):
    """None when a project file is missing; True when the learner's module gives the course's results; else False."""
    for scenario in scenarios:
        got = run_code(module_lines + scenario)
        if got.error is not None and isinstance(got.error, ModuleNotFoundError):
            return None
        want = run_code(course + "\n\n" + scenario)
        if got.error is not None or got.timeout or got.value != want.value:
            return False
    return True


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: Module and Neuron ---------------------------------------------------------------------------------

_N_HEAD = ("import random\n\nfrom nn.value import Value\n\n\nclass Module:\n    def parameters(self):\n        return []\n\n\n"
           "class Neuron(Module):\n    def __init__(self, n_inputs, relu=True):\n        self.w = []\n        for i in range(n_inputs):\n"
           "            self.w.append(Value(random.uniform(-1, 1)))\n        self.b = Value(0)\n        self.relu = relu\n\n")
_N_CALL = ("    def __call__(self, x):\n        z = self.b\n        for i in range(len(self.w)):\n            z = z + self.w[i] * x[i]\n"
           "        if self.relu:\n            return z.relu()\n        return z\n\n")
_N_PARAMS = "    def parameters(self):\n        return self.w + [self.b]"
_N_START = _N_HEAD + "    def __call__(self, x):\n        ...\n\n    def parameters(self):\n        ...\n"
_N = _N_HEAD + _N_CALL + _N_PARAMS


def _n(call=_N_CALL, params=_N_PARAMS) -> str:
    return _N_HEAD + call + params


_SET = "n.w[0].data = 0.5\nn.w[1].data = -1\nn.b.data = 1\n"
_N_SCENARIOS = [
    "n = Neuron(2)\n" + _SET + "out = n([Value(2), Value(1)])\nout.data",
    "n = Neuron(2)\n" + _SET + "out = n([Value(1), Value(4)])\nout.data",
    "n = Neuron(2, relu=False)\n" + _SET + "out = n([Value(1), Value(4)])\nout.data",
    "n = Neuron(3)\nlen(n.parameters())",
    "n = Neuron(2)\nn.w[0].data = 1\nn.w[1].data = 1\nfor p in n.parameters():\n    p.data = 2\nout = n([Value(1), Value(1)])\nout.data",
    "random.seed(1)\nn = Neuron(2)\nm = Neuron(2)\n[n.w[0].data == m.w[0].data, n.b.data]",
    "from nn.backward import backward\nn = Neuron(2)\n" + _SET + "out = n([Value(2), Value(1)])\nbackward(out)\n[n.w[0].grad, n.w[1].grad, n.b.grad]",
    "from nn.backward import backward\nn = Neuron(3)\nn.w[0].data = 1\nn.w[1].data = 2\nn.w[2].data = -1\nn.b.data = 0.5\n"
    "out = n([Value(1), Value(1), Value(1)])\nbackward(out)\n[out.data, n.w[0].grad, n.w[2].grad, n.b.grad]",
]

# --- exercise 3: Layer and MLP -------------------------------------------------------------------------------------

_M_HEAD = ("from nn.neuron import Module, Neuron\n\n\nclass Layer(Module):\n    def __init__(self, n_inputs, n_outputs, relu=True):\n"
           "        self.neurons = []\n        for i in range(n_outputs):\n            self.neurons.append(Neuron(n_inputs, relu))\n\n")
_M_LCALL = "    def __call__(self, x):\n        outputs = []\n        for n in self.neurons:\n            outputs.append(n(x))\n        return outputs\n\n"
_M_LPARAMS = ("    def parameters(self):\n        params = []\n        for n in self.neurons:\n            params = params + n.parameters()\n"
              "        return params\n\n\n")
_M_MLP_HEAD = ("class MLP(Module):\n    def __init__(self, n_inputs, sizes):\n        all_sizes = [n_inputs] + sizes\n        self.layers = []\n"
               "        for i in range(len(sizes)):\n            self.layers.append(Layer(all_sizes[i], all_sizes[i + 1], relu=i < len(sizes) - 1))\n\n")
_M_MCALL = "    def __call__(self, x):\n        for layer in self.layers:\n            x = layer(x)\n        return x\n\n"
_M_MPARAMS = "    def parameters(self):\n        params = []\n        for layer in self.layers:\n            params = params + layer.parameters()\n        return params"
_M_START = (_M_HEAD + "    def __call__(self, x):\n        ...\n\n" + _M_LPARAMS + _M_MLP_HEAD + "    def __call__(self, x):\n        ...\n\n"
            "    def parameters(self):\n        ...\n")
_M = _M_HEAD + _M_LCALL + _M_LPARAMS + _M_MLP_HEAD + _M_MCALL + _M_MPARAMS


def _m(lcall=_M_LCALL, mcall=_M_MCALL, mparams=_M_MPARAMS) -> str:
    return _M_HEAD + lcall + _M_LPARAMS + _M_MLP_HEAD + mcall + mparams


_ALL = "for p in m.parameters():\n    p.data = {v}\n"
_M_SCENARIOS = [
    "import random\nrandom.seed(0)\nfrom nn.value import Value\nm = MLP(3, [4, 4, 1])\nlen(m.parameters())",
    "import random\nrandom.seed(0)\nfrom nn.value import Value\nm = MLP(2, [2, 1])\n" + _ALL.format(v=0.5) + "out = m([Value(1), Value(2)])\n[len(out), out[0].data]",
    "import random\nrandom.seed(0)\nfrom nn.value import Value\nm = MLP(2, [2, 1])\n" + _ALL.format(v=-0.5) + "out = m([Value(1), Value(2)])\nout[0].data",
    "import random\nrandom.seed(0)\nfrom nn.value import Value\nm = MLP(2, [3, 2])\n" + _ALL.format(v=0.25) + "out = m([Value(4), Value(-1)])\n[len(out), out[0].data, out[1].data]",
    "import random\nrandom.seed(0)\nfrom nn.value import Value\nm = MLP(2, [3, 3, 1])\n" + _ALL.format(v=0.5) + "[len(m.parameters()), len(m.layers[1].parameters()), m.layers[2].neurons[0].b.data]",
    "import random\nrandom.seed(0)\nfrom nn.value import Value\nfrom nn.backward import backward\nm = MLP(2, [2, 1])\n" + _ALL.format(v=0.5)
    + "out = m([Value(1), Value(2)])[0]\nbackward(out)\ntotal = 0\nfor p in m.parameters():\n    total = total + p.grad\ntotal",
]


def neuron_ok():
    """None when neuron.py is missing; True when the learner's Neuron gives the course's results; else False."""
    if not (projects.folder() / "nn" / "neuron.py").exists():
        return None
    return _same("from nn.value import Value\nfrom nn.neuron import Module, Neuron\nimport random\n\n",
                 _VALUE + "\n\n" + _N.replace("from nn.value import Value\n", ""), [s for s in _N_SCENARIOS if "backward" not in s][:4])


def mlp_ok():
    """None when mlp.py is missing; True when the learner's Layer and MLP give the course's results; else False."""
    if not (projects.folder() / "nn" / "mlp.py").exists():
        return None
    return _same("from nn.mlp import Layer, MLP\n\n", _M, [s for s in _M_SCENARIOS if "backward" not in s])


def demo():
    """For the demo: None when mlp.py is missing; False when neuron.py or mlp.py is broken; else the number of parameters,
    the outputs for the four corners, the loss, how many parameters got a slope other than 0, and whether the backward
    was the learner's own."""
    if not (projects.folder() / "nn" / "mlp.py").exists():
        return None
    if neuron_ok() is not True or mlp_ok() is not True:
        return False
    run = run_code("import random\nfrom nn.value import Value\nfrom nn.backward import backward\nfrom nn.mlp import MLP\n\n"
                   "random.seed(3)\nm = MLP(2, [4, 4, 1])\ncorners = [(0, 0), (0, 1), (1, 0), (1, 1)]\nwanted = [0, 1, 1, 0]\n"
                   "outputs = []\nloss = Value(0)\nfor k in range(4):\n    out = m([Value(corners[k][0]), Value(corners[k][1])])[0]\n"
                   "    outputs.append(out.data)\n    r = out + Value(-wanted[k])\n    loss = loss + r * r * Value(0.25)\nbackward(loss)\n"
                   "[len(m.parameters()), outputs, loss.data, len([p for p in m.parameters() if p.grad != 0])]")
    if run.error is not None or run.timeout:
        return False
    try:
        n, outputs, loss, with_slope = run.value
        own_backward = all(module != "backward" for _, module, _, _ in projects.stand_ins())
        return int(n), [float(o) for o in outputs], float(loss), int(with_slope), own_backward
    except (TypeError, ValueError):
        return False


# The lesson is done when these are correct. "size" is the choice question in the lesson.
REQUIRED = ["size", "neuron", "mlp"]

_SCENE = "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: "
_SCENE_NL = "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: "

EXERCISES = {
    "neuron": ClassBox(
        name="neuron",
        start=_N_START,
        solution=_N,
        scenarios=_N_SCENARIOS,
        save_as=("nn", "neuron"),
        first=[_DOTS],
        mistakes=[
            Mistake("zero-start", syntax=r"^[ \t]*z\s*=\s*0\s*$",
                    en="`z` starts at the number 0, and your `Value` can only be added to another `Value`, so Python stops. Start `z` at `self.b`: that is a `Value`, and the bias term the neuron needs anyway.",
                    nl="`z` begint bij het getal 0, en je `Value` kan alleen bij een andere `Value` worden opgeteld, dus stopt Python. Laat `z` beginnen bij `self.b`: dat is een `Value`, en de bias term die het neuron toch nodig heeft."),
            Mistake("no-bias", wrong=_n(call=_N_CALL.replace("z = self.b\n", "z = Value(0)\n")),
                    en=_SCENE + "your neuron leaves out its bias term. Start `z` at `self.b`, not at 0.",
                    nl=_SCENE_NL + "je neuron laat zijn bias term weg. Begin `z` bij `self.b`, niet bij 0."),
            Mistake("always-relu", wrong=_n(call="    def __call__(self, x):\n        z = self.b\n        for i in range(len(self.w)):\n"
                                                 "            z = z + self.w[i] * x[i]\n        return z.relu()\n\n"),
                    en=_SCENE + "your neuron always uses ReLU, also when `relu` is `False`. Give back `z.relu()` only `if self.relu:`, and `z` itself otherwise.",
                    nl=_SCENE_NL + "je neuron gebruikt altijd ReLU, ook als `relu` `False` is. Geef `z.relu()` alleen terug `if self.relu:`, en anders `z` zelf."),
            Mistake("no-relu", wrong=_n(call="    def __call__(self, x):\n        z = self.b\n        for i in range(len(self.w)):\n"
                                             "            z = z + self.w[i] * x[i]\n        return z\n\n"),
                    en=_SCENE + "your neuron never uses ReLU, so below 0 it does not give 0. Give back `z.relu()` `if self.relu:`, and `z` itself otherwise.",
                    nl=_SCENE_NL + "je neuron gebruikt nooit ReLU, dus onder 0 geeft het geen 0. Geef `z.relu()` terug `if self.relu:`, en anders `z` zelf."),
            Mistake("no-b-param", wrong=_n(params="    def parameters(self):\n        return self.w"),
                    en=_SCENE + "`parameters()` leaves out the bias term. Training changes it too: `return self.w + [self.b]`.",
                    nl=_SCENE_NL + "`parameters()` laat de bias term weg. Trainen verandert die ook: `return self.w + [self.b]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: `__call__` works out `z` from `self.b`, `self.w` and `x`, with ReLU when `self.relu`; `parameters()` gives back the weights and the bias term.",
                      "nl": "Vervang de twee `...`: `__call__` rekent `z` uit met `self.b`, `self.w` en `x`, met ReLU als `self.relu`; `parameters()` geeft de weights en de bias term terug."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": _SCENE + "Does `__call__` start at `self.b`, add `self.w[i] * x[i]` for every input, and use ReLU only when `self.relu`? And does `parameters()` give back `self.w + [self.b]`?",
                      "nl": _SCENE_NL + "Begint `__call__` bij `self.b`, telt hij `self.w[i] * x[i]` op voor elke input, en gebruikt hij ReLU alleen als `self.relu`? En geeft `parameters()` `self.w + [self.b]` terug?"},
            "almost": {"en": "Right for the first neuron. But " + _SCENE[0].lower() + _SCENE[1:] + "Does your neuron use ReLU only when `self.relu` is `True`, work for any number of inputs, and give back its own `Value`s in `parameters()`?",
                       "nl": "Goed voor het eerste neuron. Maar n" + _SCENE_NL[1:] + "Gebruikt je neuron ReLU alleen als `self.relu` `True` is, werkt het voor elk aantal inputs, en geeft het zijn eigen `Value`s terug in `parameters()`?"},
            "correct": {"en": "Your `Neuron` is right in all {n} cases, also after `backward`. The check saved the box in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je `Neuron` klopt in alle {n} gevallen, ook na `backward`. De check heeft het vak in je project bewaard, als `{saved}`: oefening 3 haalt het daar binnen."},
            "kept": {"en": "Your `Neuron` is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Neuron` klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "mlp": ClassBox(
        name="mlp",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        save_as=("nn", "mlp"),
        first=[
            _DOTS,
            Mistake("no-neuron", state="empty", matches=lambda got, text: neuron_ok() is None,
                    en="Your `Neuron` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/neuron.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `Neuron` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/neuron.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-neuron", matches=lambda got, text: neuron_ok() is False,
                    en="Your code may be right, but the `Neuron` in `work/projects/nn/neuron.py` no longer works the way the check of exercise 2 expects. Did you change that file? Delete it, and check exercise 2 again.",
                    nl="Je code kan goed zijn, maar de `Neuron` in `work/projects/nn/neuron.py` werkt niet meer zoals de check van oefening 2 verwacht. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken."),
        ],
        mistakes=[
            Mistake("first-layer", wrong=_m(mcall="    def __call__(self, x):\n        return self.layers[0](x)\n\n"),
                    en=_SCENE + "your MLP gives back the output of its first layer only. Pass `x` through every layer: `for layer in self.layers:` with `x = layer(x)`.",
                    nl=_SCENE_NL + "je MLP geeft alleen de output van zijn eerste layer terug. Stuur `x` door elke layer: `for layer in self.layers:` met `x = layer(x)`."),
            Mistake("same-x", wrong=_m(mcall="    def __call__(self, x):\n        for layer in self.layers:\n            out = layer(x)\n        return out\n\n"),
                    en=_SCENE + "your MLP gives its last layer the inputs of the network, not the outputs of the layer before it. The output of a layer is the input of the next: `x = layer(x)`.",
                    nl=_SCENE_NL + "je MLP geeft zijn laatste layer de inputs van het netwerk, niet de outputs van de layer ervoor. De output van een layer is de input van de volgende: `x = layer(x)`."),
            Mistake("last-params", wrong=_m(mparams="    def parameters(self):\n        return self.layers[-1].parameters()"),
                    en=_SCENE + "`parameters()` gives back the parameters of one layer only. Put those of every layer together, as `Layer.parameters()` does for its neurons.",
                    nl=_SCENE_NL + "`parameters()` geeft de parameters van maar één layer terug. Zet die van elke layer bij elkaar, zoals `Layer.parameters()` dat doet voor zijn neuronen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: `Layer.__call__` gives back the outputs of its neurons, `MLP.__call__` passes `x` through every layer, and `MLP.parameters()` puts the parameters of its layers together.",
                      "nl": "Vervang de drie `...`: `Layer.__call__` geeft de outputs van zijn neuronen terug, `MLP.__call__` stuurt `x` door elke layer, en `MLP.parameters()` zet de parameters van zijn layers bij elkaar."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": _SCENE + "Does `Layer.__call__` give back a list with `n(x)` for every neuron, does `MLP.__call__` do `x = layer(x)` for every layer, and does `MLP.parameters()` add up the lists of all layers?",
                      "nl": _SCENE_NL + "Geeft `Layer.__call__` een lijst terug met `n(x)` voor elk neuron, doet `MLP.__call__` `x = layer(x)` voor elke layer, en telt `MLP.parameters()` de lijsten van alle layers op?"},
            "almost": {"en": "Right for the first MLP. But " + _SCENE[0].lower() + _SCENE[1:] + "Does your MLP work for any number of layers and neurons?",
                       "nl": "Goed voor het eerste MLP. Maar n" + _SCENE_NL[1:] + "Werkt je MLP voor elk aantal layers en neuronen?"},
            "correct": {"en": "Your `Layer` and `MLP` are right in all {n} cases: 41 parameters for `MLP(3, [4, 4, 1])`, outputs through every layer, and a slope for every parameter after `backward`. The check saved the box in your project, as `{saved}`.",
                        "nl": "Je `Layer` en `MLP` kloppen in alle {n} gevallen: 41 parameters voor `MLP(3, [4, 4, 1])`, outputs door elke layer heen, en een helling voor elke parameter na `backward`. De check heeft het vak in je project bewaard, als `{saved}`."},
            "kept": {"en": "Your `Layer` and `MLP` are right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Layer` en `MLP` kloppen in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("neuron", _n(call="    def __call__(self, x):\n        z = self.b\n        i = 0\n        while i < len(self.w):\n            z = z + x[i] * self.w[i]\n"
                       "            i = i + 1\n        if not self.relu:\n            return z\n        return z.relu()\n\n"), "correct", None),
    ("neuron", _n(call=_N_CALL.replace("z = self.b\n", "z = Value(0)\n")), "wrong", "no-bias"),
    ("neuron", _n(params="    def parameters(self):\n        return self.w"), "wrong", "no-b-param"),
    ("neuron", _n(params="    def parameters(self):\n        return [self.b] + self.w"), "correct", None),
    ("neuron", _n(call=_N_CALL.replace("        if self.relu:\n            return z.relu()\n", "")), "wrong", "no-relu"),
    ("neuron", _n(call="    def __call__(self, x):\n        z = 0\n        for i in range(len(self.w)):\n            z = z + self.w[i] * x[i]\n"
                       "        z = z + self.b\n        if self.relu:\n            return z.relu()\n        return z\n\n"), "error", "zero-start"),
    ("neuron", _n(params="    def parameters(self):\n        self.w.append(self.b)\n        return self.w"), "error", None),
    ("mlp", _m(mparams="    def parameters(self):\n        params = []\n        for layer in self.layers:\n            for p in layer.parameters():\n"
                       "                params.append(p)\n        return params"), "correct", None),
    ("mlp", _m(mcall="    def __call__(self, x):\n        return self.layers[0](x)\n\n"), "wrong", "first-layer"),
    ("mlp", _m(mcall="    def __call__(self, x):\n        return self.layers[-1](x)\n\n"), "wrong", "same-x"),
    ("mlp", _m(mparams="    def parameters(self):\n        params = []\n        for layer in self.layers:\n            params = layer.parameters() + params\n"
                       "        return params"), "correct", None),
]
