# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-micrograd-train: the learner's own mse, saved as nn/loss.py, and train, the training loop,
saved as nn/train.py, both on top of the learner's Value, backward and MLP, or on the course's where theirs is missing
or broken. And the pictures: an MLP learning a
curve, step by step on a slider, worked out with the course's own copy of the project; and the same with the
learner's own code."""
import functools
import re

from everything_ai import projects
from everything_ai.check import ClassBox, Mistake, run_code, same_value, with_time_limit, Timeout

# --- the course's own copy of the project so far, for the picture and to compare with ------------------------------
# The modules of the lessons before: yours when they are there and give the course's results, else the course's
# (template 1.3).
projects.builds_on(("nn", "value", "nn-micrograd-value", "value"), ("nn", "backward", "nn-micrograd-backward", "backward"),
                   ("nn", "neuron", "nn-micrograd-mlp", "neuron"), ("nn", "mlp", "nn-micrograd-mlp", "mlp"))
_VALUE, _BACKWARD, _NEURON, _MLP = (projects.course_code("nn", m) for m in ("value", "backward", "neuron", "mlp"))

# --- exercise 2: mse -------------------------------------------------------------------------------------------------

_L_HEAD = ("from nn.value import Value\n\n\ndef mse(model, xs, ys):\n    total = Value(0)\n    for k in range(len(xs)):\n"
           "        inputs = []\n        for v in xs[k]:\n            inputs.append(Value(v))\n        y_pred = model(inputs)[0]\n")
_L_BODY = "        r = y_pred + Value(-ys[k])\n        total = total + r * r\n"
_L_END = "    return total * Value(1 / len(xs))"
_L_START = _L_HEAD + "        ...\n    return ...\n"
_L = _L_HEAD + _L_BODY + _L_END


def _l(body=_L_BODY, end=_L_END) -> str:
    return _L_HEAD + body + end


# --- exercise 3: train -----------------------------------------------------------------------------------------------

_T_HEAD = ("from nn.backward import backward\nfrom nn.loss import mse\n\n\ndef train(model, xs, ys, lr, steps):\n    losses = []\n"
           "    for step in range(steps):\n        loss = mse(model, xs, ys)\n")
_T_ZERO = "        for p in model.parameters():\n            p.grad = 0\n"
_T_BACK = "        backward(loss)\n"
_T_STEP = "        for p in model.parameters():\n            p.data = p.data - lr * p.grad\n"
_T_END = "        losses.append(loss.data)\n    return losses"
_T_START = _T_HEAD + "        ...\n" + _T_END + "\n"
_T = _T_HEAD + _T_ZERO + _T_BACK + _T_STEP + _T_END


def _t(zero=_T_ZERO, back=_T_BACK, step=_T_STEP) -> str:
    return _T_HEAD + zero + back + step + _T_END


# --- the data: a curve ------------------------------------------------------------------------------------------------

XS = [round(k / 5.5 - 1, 4) for k in range(12)]
YS = [round(x * x, 4) for x in XS]
SEED, SIZES, LR, STEPS = 3, [8, 1], 0.2, 300


def _course():
    """The course's own Value, backward, MLP, mse and train, as Python objects."""
    import random
    import sys
    import types
    names = {}
    nn = types.ModuleType("nn")
    parts = [("value", _VALUE), ("backward", _BACKWARD), ("neuron", _NEURON), ("mlp", _MLP), ("loss", _L), ("train", _T)]
    # while the course's code runs, `from nn.value import Value` finds the course's own modules; afterwards every
    # name in sys.modules is what it was, so your project files are what the checks import
    saved = {key: sys.modules.get(key) for key in ["nn"] + [f"nn.{module}" for module, _ in parts]}
    sys.modules["nn"] = nn
    try:
        for module, code in parts:
            mod = types.ModuleType(f"nn.{module}")
            exec(code, mod.__dict__)
            setattr(nn, module, mod)
            sys.modules[f"nn.{module}"] = mod
            names.update(mod.__dict__)
    finally:
        for key, old in saved.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
    return names, random


@functools.lru_cache(maxsize=1)
def snapshots():
    """The course's own MLP(1, [8, 1]) on the curve: (the loss after 0, 1, …, 300 steps, the fine grid, the
    predictions on it after every 25 steps), worked out once."""
    names, random = _course()
    random.seed(SEED)
    model = names["MLP"](1, SIZES)
    Value = names["Value"]
    grid = [k / 20 - 1 for k in range(41)]

    def curve():
        return [model([Value(x)])[0].data for x in grid]

    shots, losses = {0: curve()}, []
    for done in range(25, STEPS + 1, 25):
        losses += names["train"](model, [[x] for x in XS], YS, LR, 25)
        shots[done] = curve()
    losses.append(names["mse"](model, [[x] for x in XS], YS).data)   # train keeps the loss before each step: add the last
    return losses, grid, shots


def picture(steps: int, labels: dict[str, str], losses=None, grid=None, curve=None, drag=None) -> str:
    """One chart: the 12 points of the curve and what the MLP predicts after `steps` steps. The other: the loss of every
    step, with a dot at `steps`. Without curves given, the course's own run. In a play block (L.explore), drag=1 lets
    dragging across the loss set the steps."""
    from everything_ai import chart
    if losses is None:
        losses, grid, shots = snapshots()
        curve = shots[steps]
    left = chart.Panel((-1, 1), (-0.4, 1.2), title=labels["fit"], xlabel="x", ylabel="y")
    left.points(XS, YS, "data", 4.5)
    left.line(grid, curve, "pred", 2.2)
    top = max(losses[:STEPS]) * 1.05
    right = chart.Panel((0, STEPS), (0, top), title=labels["loss"], xlabel=labels["step"], ylabel=labels["loss_y"])
    right.line(list(range(len(losses))), losses, "loss", 2)   # losses[k]: the loss after k steps
    if steps > 0:
        k = min(steps, len(losses) - 1)
        right.points([k], [losses[k]], "loss", 4.5)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=260, drag=drag)


# --- the learner's project ------------------------------------------------------------------------------------------

def _module_ok(module: str, check_lines: str, want) -> bool | None:
    if not (projects.folder() / "nn" / f"{module}.py").exists():
        return None
    got = run_code(check_lines)
    return got.error is None and not got.timeout and same_value(got.value, want)


def loss_ok():
    names, random = _course()
    random.seed(1)
    m = names["MLP"](1, [3, 1])
    want = names["mse"](m, [[0.5], [-1.0], [2.0]], [1.0, 0.0, 3.0]).data
    return _module_ok("loss", "import random\nrandom.seed(1)\nfrom nn.mlp import MLP\nfrom nn.loss import mse\nm = MLP(1, [3, 1])\n"
                              "mse(m, [[0.5], [-1.0], [2.0]], [1.0, 0.0, 3.0]).data", want)


def trained():
    """For the demo: None when train is not saved yet; False when loss.py or train.py no longer works; else (the
    losses, the fine grid, the predictions on it after training, the files that were the course's version)."""
    if not (projects.folder() / "nn" / "train.py").exists():
        return None
    if loss_ok() is not True:
        return False
    run = run_code("import random\nfrom nn.value import Value\nfrom nn.mlp import MLP\nfrom nn.train import train\n")
    if run.error is not None:
        return False
    import random
    grid = [k / 20 - 1 for k in range(41)]

    def go():
        random.seed(SEED)
        model = run.names["MLP"](1, SIZES)
        losses = run.names["train"](model, [[x] for x in XS], YS, LR, STEPS)
        Value = run.names["Value"]
        return [float(v) for v in losses], [model([Value(x)])[0].data for x in grid]
    try:
        losses, curve = with_time_limit(go, 30)
    except (Timeout, Exception):
        return False
    if len(losses) != STEPS:
        return False
    return losses, grid, curve, [f"`{module}.py`" for _, module, _, _ in projects.stand_ins()]


# The lesson is done when these are correct. "order" is the choice question in the lesson.
REQUIRED = ["order", "mse", "train"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SCENE = "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: "
_SCENE_NL = "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: "
_SEED = "import random\nrandom.seed({s})\nfrom nn.mlp import MLP\n"

def _steps_wrong(text: str) -> bool:
    """The loop does not run `steps` times: `train(..., 3)` gives back another number of losses than 3."""
    run = run_code(text + "\n\n" + _SEED.format(s=1) + "m = MLP(1, [3, 1])\nlen(train(m, [[0.5]], [1.0], 0.1, 3))")
    return run.error is None and not run.timeout and run.value != 3


_L_SCENARIOS = [
    _SEED.format(s=1) + "m = MLP(1, [3, 1])\nloss = mse(m, [[0.5], [-1.0], [2.0]], [1.0, 0.0, 3.0])\nloss.data",
    _SEED.format(s=2) + "m = MLP(2, [4, 1])\nloss = mse(m, [[1.0, 2.0], [0.0, -1.0]], [3.0, -2.0])\nloss.data",
    _SEED.format(s=1) + "m = MLP(1, [3, 1])\nfor p in m.parameters():\n    p.data = 0.5\nmse(m, [[1.0]], [2.0]).data",
    _SEED.format(s=0) + "from nn.backward import backward\nm = MLP(1, [2, 1])\nloss = mse(m, [[1.0], [2.0]], [0.0, 1.0])\nbackward(loss)\n"
                        "grads = []\nfor p in m.parameters():\n    grads.append(p.grad)\ngrads",
]
_T_SCENARIOS = [
    _SEED.format(s=1) + "m = MLP(1, [3, 1])\nlosses = train(m, [[0.5], [-1.0], [2.0]], [1.0, 0.0, 3.0], 0.05, 5)\nlosses",
    _SEED.format(s=2) + "m = MLP(2, [4, 1])\nlosses = train(m, [[1.0, 2.0], [0.0, -1.0], [1.0, 1.0]], [3.0, -2.0, 1.0], 0.02, 20)\n"
                        "[len(losses), losses[0], losses[-1]]",
    _SEED.format(s=3) + "m = MLP(1, [4, 1])\nlosses = train(m, [[0.0], [1.0]], [1.0, -1.0], 0.1, 0)\n[losses, m.parameters()[0].data]",
    _SEED.format(s=5) + "m = MLP(1, [2, 1])\ntrain(m, [[1.0], [2.0]], [2.0, 4.0], 0.01, 3)\nnumbers = []\nfor p in m.parameters():\n    numbers.append(p.data)\nnumbers",
]

EXERCISES = {
    "mse": ClassBox(
        name="mse",
        start=_L_START,
        solution=_L,
        scenarios=_L_SCENARIOS,
        digits=6,
        save_as=("nn", "loss"),
        first=[_DOTS],
        mistakes=[
            # what your Value cannot do: Python stops, and these say why
            Mistake("subtract", syntax=r"^[^#\n]*(\by_pred|[)\]])\s*-\s*[\w(]",
                    en="Your `Value` cannot subtract: it has `__add__` and `__mul__`, but no `__sub__`, so Python stops. Add the label with a minus sign instead: `y_pred + Value(-ys[k])`.",
                    nl="Je `Value` kan niet aftrekken: hij heeft `__add__` en `__mul__`, maar geen `__sub__`, dus stopt Python. Tel het label met een minteken op: `y_pred + Value(-ys[k])`."),
            Mistake("divide", syntax=r"^[^#\n]*\b(total|r|y_pred)\s*/",
                    en="Your `Value` cannot divide: it has no `__truediv__`, so Python stops. Multiply by one over the number of examples instead: `total * Value(1 / len(xs))`.",
                    nl="Je `Value` kan niet delen: hij heeft geen `__truediv__`, dus stopt Python. Vermenigvuldig in plaats daarvan met één gedeeld door het aantal voorbeelden: `total * Value(1 / len(xs))`."),
            Mistake("number", syntax=r"^[^#\n]*\*\s*\(?\s*1(\.0)?\s*/",
                    en="`1 / len(xs)` is a plain number, and your `Value` can only be multiplied by another `Value`, so Python stops. Make it a `Value` first: `total * Value(1 / len(xs))`.",
                    nl="`1 / len(xs)` is een gewoon getal, en je `Value` kan alleen met een andere `Value` vermenigvuldigd worden, dus stopt Python. Maak er eerst een `Value` van: `total * Value(1 / len(xs))`."),
            Mistake("power", syntax=r"\*\*",
                    en="Your `Value` has no `**`, so Python stops. Square the residual by multiplying it by itself: `r * r`.",
                    nl="Je `Value` heeft geen `**`, dus stopt Python. Kwadrateer het residu door het met zichzelf te vermenigvuldigen: `r * r`."),
            Mistake("no-graph", wrong=_l(end="    return Value(total.data * (1 / len(xs)))"),
                    en=_SCENE + "the loss is right, but `backward` finds no slopes in it. A new `Value` made from a `.data` does not know where its number came from, so `backward` cannot reach the parameters. Keep everything a `Value`: `total * Value(1 / len(xs))`.",
                    nl=_SCENE_NL + "de loss klopt, maar `backward` vindt er geen hellingen in. Een nieuwe `Value` gemaakt van een `.data` weet niet waar zijn getal vandaan komt, dus `backward` komt niet bij de parameters. Houd alles een `Value`: `total * Value(1 / len(xs))`."),
            Mistake("no-square", wrong=_l(body="        r = y_pred + Value(-ys[k])\n        total = total + r\n"),
                    en=_SCENE + "you add up the residuals themselves. A residual of +1 and one of −1 would cancel out. Add up their squares: `r * r`.",
                    nl=_SCENE_NL + "je telt de residuen zelf op. Een residu van +1 en een van −1 zouden elkaar opheffen. Tel hun kwadraten op: `r * r`."),
            Mistake("no-mean", wrong=_l(end="    return total"),
                    en=_SCENE + "that is the sum of the squared residuals. The mean squared error divides by the number of examples: `total * Value(1 / len(xs))`.",
                    nl=_SCENE_NL + "dat is de som van de gekwadrateerde residuen. De mean squared error deelt door het aantal voorbeelden: `total * Value(1 / len(xs))`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in the loop, add the square of the residual to `total`; after it, give back the mean.",
                      "nl": "Vervang de twee `...`: tel in de loop het kwadraat van het residu op bij `total`; geef daarna het gemiddelde terug."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": _SCENE + "Does the loop add `r * r`, with `r = y_pred + Value(-ys[k])`, and does `mse` give back `total * Value(1 / len(xs))`?",
                      "nl": _SCENE_NL + "Telt de loop `r * r` op, met `r = y_pred + Value(-ys[k])`, en geeft `mse` `total * Value(1 / len(xs))` terug?"},
            "almost": {"en": "Right in the first case. But a" + _SCENE[1:] + "Does `mse` work for any number of examples and features, and does everything in it stay a `Value`, so that `backward` reaches every parameter?",
                       "nl": "Goed in het eerste geval. Maar n" + _SCENE_NL[1:] + "Werkt `mse` voor elk aantal voorbeelden en features, en blijft alles erin een `Value`, zodat `backward` elke parameter bereikt?"},
            "correct": {"en": "Your `mse` is right in all {n} cases, and `backward` gets every slope from it. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je `mse` klopt in alle {n} gevallen, en `backward` haalt er elke helling uit. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your `mse` is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `mse` klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "train": ClassBox(
        name="train",
        start=_T_START,
        solution=_T,
        scenarios=_T_SCENARIOS,
        digits=6,
        save_as=("nn", "train"),
        first=[
            _DOTS,
            Mistake("no-loss", state="empty", matches=lambda got, text: loss_ok() is None,
                    en="Your `mse` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/loss.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `mse` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/loss.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-loss", matches=lambda got, text: loss_ok() is False,
                    en="Your loop may be right, but the `mse` in `work/projects/nn/loss.py` no longer gives the course's loss. Did you change that file? Delete it, and check exercise 2 again.",
                    nl="Je loop kan goed zijn, maar de `mse` in `work/projects/nn/loss.py` geeft niet meer de loss van de course. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken."),
        ],
        mistakes=[
            Mistake("no-zero", wrong=_t(zero=""),
                    en=_SCENE + "the slopes are never set back to 0. `backward_step()` adds to `grad`, so every step moves by the slopes of all the steps before it too. Set every `p.grad = 0` before `backward(loss)`.",
                    nl=_SCENE_NL + "de hellingen worden nooit terug op 0 gezet. `backward_step()` telt op bij `grad`, dus elke stap schuift ook met de hellingen van alle stappen ervoor. Zet elke `p.grad = 0` vóór `backward(loss)`."),
            # also a loop without backward(loss), without a step, or with the step before backward: the same outcome
            Mistake("zero-late", wrong=_t(zero="", step="        for p in model.parameters():\n            p.grad = 0\n"
                                                    "        for p in model.parameters():\n            p.data = p.data - lr * p.grad\n"),
                    en=_SCENE + "the loss never changes, so the parameters never move. Each round needs all three parts, in this order: every `p.grad = 0`, then `backward(loss)`, then `p.data = p.data - lr * p.grad` for every parameter. Grads set to 0 between `backward(loss)` and the step leave the step nothing to move by.",
                    nl=_SCENE_NL + "de loss verandert nooit, dus de parameters schuiven nooit. Elke ronde heeft alle drie de delen nodig, in deze volgorde: elke `p.grad = 0`, dan `backward(loss)`, dan `p.data = p.data - lr * p.grad` voor elke parameter. Grads die tussen `backward(loss)` en de stap op 0 gezet worden, laten de stap niets om mee te schuiven."),
            Mistake("uphill", wrong=_t(step="        for p in model.parameters():\n            p.data = p.data + lr * p.grad\n"),
                    en=_SCENE + "the loss goes up: each step moves with the slope, uphill. Move against it: `p.data - lr * p.grad`.",
                    nl=_SCENE_NL + "de loss gaat omhoog: elke stap schuift met de helling mee, bergop. Schuif ertegenin: `p.data - lr * p.grad`."),
            # a loop that does not run `steps` times: say so, instead of showing a long list of losses
            Mistake("steps", matches=lambda got, text: _steps_wrong(text),
                    en="Your loop does not run exactly `steps` times, so `train` gives back another number of losses than it should. Keep the line `for step in range(steps):` as it was: one round, and one loss, per step.",
                    nl="Je loop draait niet precies `steps` keer, dus `train` geeft een ander aantal losses terug dan zou moeten. Laat de regel `for step in range(steps):` zoals hij was: één ronde, en één loss, per stap."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with three parts of the loop: every `grad` to 0, `backward(loss)`, and a step against the slope for every parameter.",
                      "nl": "Vervang de `...` door drie delen van de loop: elke `grad` op 0, `backward(loss)`, en een stap tegen de helling in voor elke parameter."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does the loop run `steps` times, with one `backward(loss)` per step?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Draait de loop `steps` keer, met één `backward(loss)` per stap?"},
            "wrong": {"en": _SCENE + "Does each step set every `p.grad` to 0, then call `backward(loss)`, and then do `p.data = p.data - lr * p.grad` for every parameter?",
                      "nl": _SCENE_NL + "Zet elke stap elke `p.grad` op 0, roept hij dan `backward(loss)` aan, en doet hij daarna `p.data = p.data - lr * p.grad` voor elke parameter?"},
            "almost": {"en": "Right for the first run. But a" + _SCENE[1:] + "Does your loop work for any number of steps, also 0, and change every parameter of the model?",
                       "nl": "Goed voor de eerste run. Maar n" + _SCENE_NL[1:] + "Werkt je loop voor elk aantal stappen, ook 0, en verandert hij elke parameter van het model?"},
            "correct": {"en": "Your `train` is right in all {n} cases: the same losses as the course's, step by step. The check saved it in your project, as `{saved}`.",
                        "nl": "Je `train` klopt in alle {n} gevallen: dezelfde losses als die van de course, stap voor stap. De check heeft hem in je project bewaard, als `{saved}`."},
            "kept": {"en": "Your `train` is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `train` klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("mse", _l(body="        total = total + (y_pred + Value(-ys[k])) * (y_pred + Value(-ys[k]))\n"), "correct", None),
    ("mse", _l(body="        r = y_pred + Value(-ys[k])\n        total = total + r\n"), "wrong", "no-square"),
    ("mse", _l(end="    return total"), "wrong", "no-mean"),
    ("mse", _l(body="        r = y_pred - ys[k]\n        total = total + r * r\n"), "error", "subtract"),
    ("mse", _l(end="    return total / len(xs)"), "error", "divide"),
    ("mse", _l(end="    return total * (1 / len(xs))"), "error", "number"),
    ("mse", _l(body="        r = y_pred + Value(-ys[k])\n        total = total + r ** 2\n"), "error", "power"),
    ("mse", _l(end="    return Value(total.data / len(xs))"), "wrong", "no-graph"),
    ("mse", _l(body="        r = Value(-ys[k]) + y_pred\n        total = total + r * r * Value(1 / len(xs))\n", end="    return total"), "correct", None),
    ("train", _t(zero="        params = model.parameters()\n        for p in params:\n            p.grad = 0\n"), "correct", None),
    ("train", _t(zero=""), "wrong", "no-zero"),
    ("train", _t(back=""), "wrong", "zero-late"),
    ("train", _t(back="", step=_T_STEP + _T_BACK), "wrong", "zero-late"),
    ("train", _t(zero="", step=_T_STEP + _T_ZERO), "correct", None),
    ("train", _T.replace("range(steps)", "range(300)"), "wrong", "steps"),
    ("train", _T.replace("range(steps)", "range(steps + 1)"), "wrong", "steps"),
    ("train", _T.replace("range(steps)", "range(0, steps)").replace("            p.grad = 0\n", "            pass\n"), "wrong", "no-zero"),
    ("train", _t(step="        for p in model.parameters():\n            p.data = p.data + lr * p.grad\n"), "wrong", "uphill"),
]
