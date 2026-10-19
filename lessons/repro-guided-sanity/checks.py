# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-sanity: overfit_one_batch, which trains a GPT on one batch of equations and
gives the loss of every step, saved as repro/overfit.py; best_loss, the lowest loss a model can reach that sees only
a; and sanity, the three sanity checks of the reproduction kit, saved as repro/sanity.py. The picture shows two
recorded runs of the course's ModernGPT on one batch, one of them with a bug; the demo runs the learner's sanity check
on their ModernGPT, and on the same model with that bug."""
import functools
import importlib.util
import math
import re

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, Timeout, _show_rounded, run_code, with_time_limit


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _checks("repro-guided-build", "everything_ai_checks_repro_guided_build_for_sanity")
PAPER, FRACTION, tokens, number = B.PAPER, B.FRACTION, B.tokens, B.number
# The kit so far, and the model it adapts: where a module is missing or broken, the course's version stands in.
BUILDS_ON = B.BUILDS_ON + (("repro", "data", "repro-guided-build", "dataset"), ("repro", "settings", "repro-guided-build", "settings"),
                           ("repro", "overfit", "repro-guided-sanity", "overfit"))
projects.builds_on(*BUILDS_ON)

VOCAB = PAPER["p"] + 2   # the numbers 0 to p - 1, the operation and the sign =
BATCH = PAPER["batch"]   # the paper's batch: the sanity checks use one batch of the size of the real run
STEPS = 150              # steps of learning one batch by heart
LR = PAPER["lr"]
FIRST_TOLERANCE = 0.5    # the first loss is right when it is this close to ln VOCAB
LEARNED = 0.05           # the batch is learned when the loss of the last step is below this
# The full run: the step at which validation reached 99% in the course's run at FRACTION (repro-guided-build), and
# the paper's budget; both at the seconds per step measured there.
RUN_STEPS, BUDGET, SECONDS_PER_STEP = B.FRACTIONS[FRACTION]["v"], PAPER["steps"], B.SECONDS_PER_STEP

# --- measured: the course's ModernGPT with the paper's settings learning the first 512 training equations by heart,
# AdamW at the paper's learning rate and betas without weight decay, seed 0, one thread, the loss before each step
# (data/tools/measure_sanity.py); "right" reads its guess at the place of =, "wrong" at the place of a ---
RUNS = {
    "right": [
        4.7606, 4.6099, 4.5232, 4.4702, 4.4226, 4.3707, 4.3097, 4.2358, 4.1449, 4.0341, 3.902, 3.7498, 3.5819, 3.4078,
        3.2405, 3.0885, 2.9502, 2.8201, 2.6949, 2.5689, 2.4372, 2.3034, 2.1711, 2.0406, 1.9116, 1.7846, 1.6584,
        1.5313, 1.4029, 1.2753, 1.1522, 1.0361, 0.9265, 0.8212, 0.7208, 0.6275, 0.5432, 0.4689, 0.4063, 0.3528,
        0.3033, 0.2541, 0.2248, 0.1864, 0.1657, 0.1399, 0.1251, 0.1077, 0.0951, 0.0856, 0.0747, 0.0675, 0.0615,
        0.0547, 0.05, 0.0462, 0.0421, 0.0385, 0.036, 0.0335, 0.0309, 0.0289, 0.0273, 0.0256, 0.024, 0.0227, 0.0217,
        0.0206, 0.0195, 0.0186, 0.0179, 0.0172, 0.0165, 0.0158, 0.0152, 0.0147, 0.0142, 0.0137, 0.0133, 0.0129,
        0.0126, 0.0122, 0.0119, 0.0115, 0.0113, 0.011, 0.0107, 0.0105, 0.0102, 0.01, 0.0098, 0.0096, 0.0094, 0.0092,
        0.009, 0.0089, 0.0087, 0.0086, 0.0084, 0.0083, 0.0081, 0.008, 0.0079, 0.0077, 0.0076, 0.0075, 0.0074, 0.0073,
        0.0071, 0.007, 0.0069, 0.0068, 0.0067, 0.0066, 0.0065, 0.0065, 0.0064, 0.0063, 0.0062, 0.0061, 0.006, 0.0059,
        0.0059, 0.0058, 0.0057, 0.0056, 0.0056, 0.0055, 0.0054, 0.0053, 0.0053, 0.0052, 0.0051, 0.0051, 0.005, 0.0049,
        0.0049, 0.0048, 0.0048, 0.0047, 0.0046, 0.0046, 0.0045, 0.0045, 0.0044, 0.0044, 0.0043, 0.0043, 0.0042,
        0.0042, 0.0041
    ],
    "wrong": [
        4.7942, 4.6273, 4.4655, 4.3086, 4.1564, 4.0089, 3.8662, 3.7285, 3.5961, 3.4695, 3.3488, 3.2344, 3.1266,
        3.0254, 2.9309, 2.843, 2.7615, 2.6863, 2.617, 2.5533, 2.495, 2.4417, 2.3931, 2.3487, 2.3084, 2.2718, 2.2386,
        2.2085, 2.1812, 2.1561, 2.1331, 2.1118, 2.092, 2.0737, 2.0568, 2.0412, 2.0269, 2.0136, 2.0012, 1.9896, 1.9787,
        1.9685, 1.959, 1.95, 1.9416, 1.9338, 1.9264, 1.9196, 1.913, 1.9069, 1.901, 1.8955, 1.8904, 1.8856, 1.881,
        1.8767, 1.8725, 1.8686, 1.865, 1.8615, 1.8582, 1.855, 1.852, 1.8492, 1.8465, 1.8439, 1.8414, 1.8391, 1.8369,
        1.8347, 1.8327, 1.8307, 1.8289, 1.8271, 1.8253, 1.8237, 1.8221, 1.8206, 1.8192, 1.8178, 1.8164, 1.8151,
        1.8139, 1.8127, 1.8115, 1.8104, 1.8093, 1.8083, 1.8072, 1.8063, 1.8053, 1.8044, 1.8035, 1.8027, 1.8018, 1.801,
        1.8002, 1.7995, 1.7987, 1.798, 1.7973, 1.7967, 1.796, 1.7954, 1.7947, 1.7941, 1.7935, 1.793, 1.7924, 1.7919,
        1.7913, 1.7908, 1.7903, 1.7898, 1.7893, 1.7889, 1.7884, 1.788, 1.7875, 1.7871, 1.7867, 1.7863, 1.7859, 1.7855,
        1.7851, 1.7847, 1.7844, 1.784, 1.7836, 1.7833, 1.783, 1.7826, 1.7823, 1.782, 1.7817, 1.7814, 1.7811, 1.7808,
        1.7805, 1.7802, 1.78, 1.7797, 1.7794, 1.7792, 1.7789, 1.7787, 1.7784, 1.7782, 1.7779, 1.7777, 1.7775
    ],
}
# --- end of measured ---


@functools.cache
def batch() -> tuple:
    """The batch of the sanity checks: the first BATCH training equations of the paper's data at FRACTION, seed 0."""
    train, _ = run_code(B.EXERCISES["dataset"].solution).names["dataset"](PAPER["p"], FRACTION, 0)
    return tuple(train[:BATCH])


def best(rows) -> float:
    """The lowest loss a model can reach on rows when it sees only a: the course's best_loss."""
    return run_code(_BEST).names["best_loss"](list(rows))


@functools.cache
def floor() -> float:
    """best() of the batch of the sanity checks."""
    return best(batch())


def reads_at_a(model):
    """The bug of the lesson, planted in a model that works: the logits at its last place are those of its first
    place, so it guesses c where it has seen only a."""
    import torch

    class ReadsAtA(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            return torch.cat([out[:, :-1], out[:, :1]], dim=1)

    return ReadsAtA(model)


def course_run(which: str, steps: int = STEPS + 1) -> list[float]:
    """A run of RUNS, made again: the course's ModernGPT, and for "wrong" with the bug planted, learning batch() with
    the course's overfit_one_batch; the loss before each of `steps` steps."""
    import torch
    names = run_code(projects.course_code("llm", "modern_gpt")).names
    overfit = run_code(_O).names["overfit_one_batch"]
    x, y = tokens(list(batch()), PAPER["p"])
    torch.manual_seed(0)
    model = names["ModernGPT"](VOCAB, PAPER["width"], PAPER["heads"], PAPER["heads"], PAPER["layers"])
    if which == "wrong":
        model = reads_at_a(model)
    return overfit(model, x, y, steps, LR)


def picture(step: int, labels: dict) -> str:
    """The two recorded runs over their steps, with a dot on each at `step`, and the loss of guessing evenly, ln VOCAB.
    Both curves are losses, so both purple: the run with the bug dashed. No line from the axis up to the dots: below
    the dashed curve it would cross the label of the other run. Drawn as SVG text, once for every stop of the slider
    (L.explore); dragging across it sets the step."""
    from everything_ai import chart
    right, wrong = RUNS["right"], RUNS["wrong"]
    xs = list(range(len(right)))
    p = chart.Panel((0, STEPS), (0, 5.2), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], yticks=[0, 1, 2, 3, 4, 5])
    even = math.log(VOCAB)
    p.line([0, STEPS], [even, even], "ink-3", 1.2, dashed=True)
    p.text(STEPS - 2, even + 0.3, labels["even"], "ink-2", anchor="end")   # the g's hang clear of the dashed line
    p.line(xs, wrong, "loss", 2, dashed=True)
    p.line(xs, right, "loss", 2.4)
    p.points([step, step], [right[step], wrong[step]], "loss", 4.5)
    p.text(STEPS - 2, wrong[-1] + 0.42, labels["wrong"], "loss", anchor="end")   # clear of the dots below them
    p.text(STEPS - 2, right[-1] + 0.42, labels["right"], "loss", anchor="end")
    return chart.figure([p], alt=labels["alt"], width=460, height=300, drag=0)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_KEEP = B._KEEP


def _close(got, want) -> bool:
    """Numbers, also inside lists, equal to 4 decimals; anything else equal."""
    if isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(got, bool) or isinstance(want, bool):
        return got == want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        return abs(got - want) <= 1e-4 + 1e-4 * abs(want)
    return got == want


# --- exercise 2: overfit_one_batch ----------------------------------------------------------------------------------------

_O_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef overfit_one_batch(model, x, y, steps, lr):\n"
           "    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)\n"
           "    losses = []\n    for step in range(steps):\n")
_O_TAIL = "    return losses\n"
_O_START = (_O_HEAD + "        # the loss on the answers: the logits at the last place, against y; keep it in losses\n"
            "        # then the gradients back to 0, backward, and a step of opt\n        ...\n" + _O_TAIL)
_O_LOSS = "        loss = F.cross_entropy(model(x)[:, -1], y)\n"
_O_KEEP = "        losses.append(loss.item())\n"
_O_STEP = "        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"


def _o(loss: str = _O_LOSS, keep: str = _O_KEEP, step: str = _O_STEP) -> str:
    return _O_HEAD + loss + keep + step + _O_TAIL


_O = _o()
_O_DATA = ("import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nx = torch.randint(0, 7, (12, 4))\ny = torch.randint(0, 7, (12,))\n"
           "model = nn.Sequential(nn.Embedding(7, 8), nn.Linear(8, 7))\n")
_O_SCENARIOS = [
    # learning the batch: the first and the last loss, and how many
    _O_DATA + "losses = overfit_one_batch(model, x, y, 40, 0.05)\n[round(losses[0], 4), round(losses[-1], 4), len(losses)]",
    # one step: the loss before it, and what the step did to the model
    _O_DATA + "losses = overfit_one_batch(model, x, y, 1, 0.05)\n[[round(v, 4) for v in losses], [round(v, 4) for v in model[1].bias.tolist()]]",
    # the model itself has learned: its loss afterwards, measured again
    _O_DATA + "overfit_one_batch(model, x, y, 25, 0.05)\nround(nn.functional.cross_entropy(model(x)[:, -1], y).item(), 4)",
]
_O_EN, _O_NL = "Your `overfit_one_batch` gives other losses than the course's", "Je `overfit_one_batch` geeft andere losses dan die van de course"

# --- exercise 1: best_loss ------------------------------------------------------------------------------------------------

_BEST_HEAD = "import math\n\n\ndef best_loss(rows):\n    counts = {}\n    for a, b, c in rows:\n"
_BEST_COUNT = "        counts[a] = counts.get(a, 0) + 1\n"
_BEST_MID = "    total = 0.0\n    for a in counts:\n"
_BEST_SUM = "        total = total + counts[a] * math.log(counts[a])\n"
_BEST_TAIL = "    return total / len(rows)\n"
_BEST_START = (_BEST_HEAD + "        # count the equations with this a\n        ...\n" + _BEST_MID
               + "        # each of the counts[a] equations with this a costs ln counts[a] at best\n        ...\n" + _BEST_TAIL)


def _best(count: str = _BEST_COUNT, add: str = _BEST_SUM, tail: str = _BEST_TAIL) -> str:
    return _BEST_HEAD + count + _BEST_MID + add + tail


_BEST = _best()
_ROWS_LESSON = [(3, 0, 3), (3, 1, 4), (0, 2, 2), (0, 0, 0), (3, 4, 2)]   # the batch the lesson's text names

# --- exercise 4: sanity ---------------------------------------------------------------------------------------------------

_S_HEAD = ("import math\n\nimport torch\nimport torch.nn.functional as F\n\nfrom repro.overfit import overfit_one_batch\n\n\n"
           "def sanity(model, x, y, vocab, steps):\n    with torch.no_grad():\n        logits = model(x)\n")
_S_SHAPE = "    shape = tuple(logits.shape) == (x.shape[0], x.shape[1], vocab)\n"
_S_FIRST = "    first = F.cross_entropy(logits[:, -1], y).item()\n"
_S_LOSSES = "    losses = overfit_one_batch(model, x, y, steps, 1e-3)\n"
_S_TAIL = ('    return {"shape": shape, "first": abs(first - math.log(vocab)) < 0.5, "learns": losses[-1] < 0.05,\n'
           '            "first_loss": first, "last_loss": losses[-1]}\n')
_S_START = (_S_HEAD + "    # 1. shape: True when the logits are len(x) by places by vocab\n"
            "    # 2. first: the loss on the answers before any training\n"
            "    # 3. losses: learning the batch by heart, steps steps at a learning rate of 1e-3\n    ...\n" + _S_TAIL)


def _s(shape: str = _S_SHAPE, first: str = _S_FIRST, losses: str = _S_LOSSES, tail: str = _S_TAIL) -> str:
    return _S_HEAD + shape + first + losses + tail


_S = _s()
_S_DATA = ("import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nx = torch.randint(0, 7, (12, 4))\ny = (3 * x[:, -1]) % 7\n"
           "model = nn.Sequential(nn.Embedding(7, 8), nn.Linear(8, 7))\n")   # y follows from the last token: a batch it can learn
_S_ROUND = "\n{k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items()}"
_S_SCENARIOS = [
    # a model that works: every check passes (a small model at the paper's learning rate needs more steps)
    _S_DATA + "report = sanity(model, x, y, 7, 800)" + _S_ROUND,
    # the vocabulary the check is told is not the model's: the shape, and the first loss against ln 9
    _S_DATA + "report = sanity(model, x, y, 9, 5)" + _S_ROUND,
    # weights that start much too large: a first loss far above ln 7
    _S_DATA + "with torch.no_grad():\n    model[1].weight.mul_(30)\nreport = sanity(model, x, y, 7, 5)" + _S_ROUND,
    # a model that loses a place: the shape is wrong, though its last number is the vocabulary
    _S_DATA + "class LosesAPlace(nn.Module):\n    def __init__(self, inner):\n        super().__init__()\n        self.inner = inner\n\n"
              "    def forward(self, x):\n        return self.inner(x)[:, :-1]\n\n\nreport = sanity(LosesAPlace(model), x, y, 7, 5)" + _S_ROUND,
    # an answer that leaks into the input: a first loss far below ln 7
    _S_DATA + "class Leaks(nn.Module):\n    def __init__(self, inner):\n        super().__init__()\n        self.inner = inner\n\n"
              "    def forward(self, x):\n        return self.inner(x) + 20 * nn.functional.one_hot(3 * x % 7, 7).float()\n\n\n"
              "report = sanity(Leaks(model), x, y, 7, 5)" + _S_ROUND,
    # a model that cannot learn: its parameters are frozen
    _S_DATA + "for t in model.parameters():\n    t.requires_grad_(False)\nmodel[1].bias.requires_grad_(True)\n"
              "report = sanity(model, x, y, 7, 20)" + _S_ROUND,
]
_S_EN, _S_NL = "Your `sanity` gives another report than the course's", "Je `sanity` geeft een ander rapport dan die van de course"


def demo(steps: int = STEPS):
    """The learner's sanity check of their project, on their ModernGPT with their data and settings: the first batch
    of the training data, and the same model with the bug planted. None when repro/overfit.py or repro/sanity.py is not
    in the project yet; "sanity" when it stops with an error, "slow" when it takes more than three minutes for one model,
    "model" when the model or the data does. Where a module it uses is missing or broken, the course's version stands
    in, and "stand_ins" names it."""
    folder = projects.folder() / "repro"
    if not (folder / "overfit.py").is_file() or not (folder / "sanity.py").is_file():
        return None
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    try:
        return _demo(stand, steps)
    finally:
        projects.forget()


def _demo(stand, steps: int):
    import importlib
    import torch
    try:
        sanity = importlib.import_module("repro.sanity").sanity
    except Exception:
        return "sanity"
    try:
        s = importlib.import_module("repro.settings").SETTINGS
        train, _ = importlib.import_module("repro.data").dataset(s["p"], s["fraction"], 0)
        rows = train[:min(s["batch"], len(train))]
        x, y = tokens(rows, s["p"])
        ModernGPT = importlib.import_module("llm.modern_gpt").ModernGPT
        vocab = s["p"] + 2

        def model():
            torch.manual_seed(0)
            return ModernGPT(vocab, s["width"], s["heads"], s["heads"], s["layers"])

        model().forward(x[:2])
    except Exception:
        return "model"
    reports = {}
    try:
        for which in ("yours", "planted"):
            m = model() if which == "yours" else reads_at_a(model())
            reports[which] = with_time_limit(lambda m=m: sanity(m, x, y, vocab, steps), 180)   # about 20 s each here
    except Timeout:
        return "slow"
    except Exception:
        return "sanity"
    return {"reports": reports, "batch": len(rows), "vocab": vocab, "even": math.log(vocab), "floor": best(rows),
            "steps": steps, "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}


EXERCISES = {
    "overfit": ClassBox(
        name="overfit",
        start=_O_START,
        solution=_O,
        scenarios=_O_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("repro", "overfit"),
        first=[_DOTS],
        mistakes=[
            Mistake("first-place", wrong=_o(loss=_O_LOSS.replace("[:, -1]", "[:, 0]")),
                    en=_O_EN + ": your loss uses the logits of the first place, `[:, 0]`, where the model has seen only `a`. Its guess for `c` is at the last place, that of `=`: `model(x)[:, -1]`.",
                    nl=_O_NL + ": je loss gebruikt de logits van de eerste plek, `[:, 0]`, waar het model alleen `a` heeft gezien. Zijn gok voor `c` staat op de laatste plek, die van `=`: `model(x)[:, -1]`."),
            Mistake("no-zero", wrong=_o(step="        loss.backward()\n        opt.step()\n"),
                    en=_O_EN + ": the gradients are never set back to 0, so every step also takes all gradients before it. Call `opt.zero_grad()` before `loss.backward()`.",
                    nl=_O_NL + ": de gradients gaan nooit terug naar 0, dus elke stap neemt ook alle gradients ervoor mee. Roep `opt.zero_grad()` aan vóór `loss.backward()`."),
            Mistake("no-step", wrong=_o(step="        opt.zero_grad()\n        loss.backward()\n"),
                    en=_O_EN + ": the model never changes, so the loss stays where it started. `opt.step()` comes after `loss.backward()`, with nothing between them that sets the gradients back to 0.",
                    nl=_O_NL + ": het model verandert nooit, dus de loss blijft waar hij begon. `opt.step()` komt na `loss.backward()`, zonder iets ertussen dat de gradients terug naar 0 zet."),
            Mistake("keep-tensor", error=lambda e: isinstance(e, TypeError) and "Tensor" in str(e) and "__round__" in str(e),
                    en="Your `losses` holds tensors, not numbers: each `loss` is a tensor, with the whole computation behind it. Keep the number in it: `loss.item()`.",
                    nl="Je `losses` bevat tensors, geen getallen: elke `loss` is een tensor, met de hele berekening erachter. Bewaar het getal erin: `loss.item()`."),
            Mistake("keep-after", wrong=_O_HEAD + _O_LOSS + _O_STEP + "        losses.append(F.cross_entropy(model(x)[:, -1], y).item())\n" + _O_TAIL,
                    en=_O_EN + ": each loss in your list is measured after its step. Keep the loss you take the step with, before the update: `losses.append(loss.item())` right after computing it.",
                    nl=_O_NL + ": elke loss in je lijst is na zijn stap gemeten. Bewaar de loss waarmee je de stap zet, vóór de update: `losses.append(loss.item())` meteen nadat je hem berekent."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...`: the loss of the logits at the last place against `y`, kept in `losses` as a number, then the three lines of a training step.",
                      "nl": "Vervang de `...`: de loss van de logits op de laatste plek tegen `y`, als getal bewaard in `losses`, en dan de drie regels van een trainingsstap."},
            "wrong": {"en": _O_EN + ". Does every round take the loss on the same `x` and `y` at the last place, keep it with `.item()`, set the gradients to 0, and take a step after `backward()`?",
                      "nl": _O_NL + ". Neemt elke ronde de loss op dezelfde `x` en `y` op de laatste plek, bewaart hij die met `.item()`, zet hij de gradients op 0, en zet hij een stap na `backward()`?"},
            "almost": {"en": "Your `overfit_one_batch` gives the course's losses for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Is the line with the optimizer as it was given, without weight decay, and does every step use the loss you keep?",
                       "nl": "Je `overfit_one_batch` geeft de losses van de course voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Staat de regel met de optimizer er nog zoals hij gegeven was, zonder weight decay, en gebruikt elke stap de loss die je bewaart?"},
            "correct": {"en": "Your `overfit_one_batch` trains a GPT on one batch and keeps the loss of every step. It is now in your project, as `work/projects/repro/overfit.py`.",
                        "nl": "Je `overfit_one_batch` traint een GPT op één batch en bewaart de loss van elke stap. Hij staat nu in je project, als `work/projects/repro/overfit.py`."},
            "kept": {"en": "Your `overfit_one_batch` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `overfit_one_batch` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "best-loss": FunctionBox(
        name="best-loss",
        function="best_loss",
        start=_BEST_START,
        solution=_BEST,
        visible=[(_ROWS_LESSON,)],
        show=lambda v: _show_rounded(v, 4),   # 0.9364, not 0.936426245424844
        hidden=[([(0, 1, 1)],), ([(3, 0, 3), (3, 1, 4), (3, 2, 0), (3, 3, 1)],), ([(0, 0, 0), (1, 0, 1), (2, 0, 2)],),
                ([(a, b, (a + b) % 5) for a in range(5) for b in range(5)],), ([(1, 1, 2), (1, 2, 3), (2, 1, 3), (2, 2, 4), (2, 3, 0), (4, 4, 3)],)],
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("ln-n", wrong=_best(add="        total = total + counts[a] * math.log(len(rows))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take ln of the number of all equations. A model that sees `a` only has to choose among the answers of that `a`: ln `counts[a]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt ln van het aantal van alle vergelijkingen. Een model dat `a` ziet, hoeft alleen te kiezen tussen de antwoorden van die `a`: ln `counts[a]`."),
            Mistake("once-per-a", wrong=_best(add="        total = total + math.log(counts[a])\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add ln `counts[a]` once for each `a`. Each of the `counts[a]` equations with that `a` costs it: `counts[a] * math.log(counts[a])`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt ln `counts[a]` één keer per `a` op. Elk van de `counts[a]` vergelijkingen met die `a` kost het: `counts[a] * math.log(counts[a])`."),
            Mistake("no-mean", wrong=_best(tail="    return total\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sum over all equations. The loss is a mean: divide by `len(rows)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de som over alle vergelijkingen. De loss is een gemiddelde: deel door `len(rows)`."),
            Mistake("count-c", wrong=_best(count="        counts[c] = counts.get(c, 0) + 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count the equations per answer `c`. The model sees `a`, so count per `a`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de vergelijkingen per antwoord `c`. Het model ziet `a`, dus tel per `a`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace each `...`: first count in `counts` how many equations each `a` has, then add to `total` what the equations of each `a` cost at best.",
                      "nl": "Vervang elke `...`: tel eerst in `counts` hoeveel vergelijkingen elke `a` heeft, en tel dan bij `total` op wat de vergelijkingen van elke `a` op zijn best kosten."},
            "arguments": {"en": "The check calls `{call}`, with one list of equations, and the `def` line of your `best_loss` has another number of names between its parentheses. Keep `(rows)`.",
                          "nl": "De check roept `{call}` aan, met één lijst vergelijkingen, en de `def`-regel van je `best_loss` heeft een ander aantal namen tussen de haakjes. Houd `(rows)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you count the equations per `a`, does each `a` add its count times ln of its count, and do you divide by the number of equations?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je de vergelijkingen per `a`, telt elke `a` zijn aantal keer ln van zijn aantal op, en deel je door het aantal vergelijkingen?"},
            "correct": {"en": "Your `best_loss` gives the lowest loss a model can reach on a batch when it sees only `a`: 0 when every `a` comes once, and more the more answers an `a` has.",
                        "nl": "Je `best_loss` geeft de laagste loss die een model op een batch kan halen als het alleen `a` ziet: 0 als elke `a` één keer voorkomt, en meer naarmate een `a` meer antwoorden heeft."},
        },
    ),
    "sanity": ClassBox(
        name="sanity",
        start=_S_START,
        solution=_S,
        scenarios=_S_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("repro", "sanity"),
        first=[_DOTS],
        mistakes=[
            Mistake("first-trained", wrong=_S_HEAD + _S_SHAPE + _S_LOSSES + "    first = F.cross_entropy(model(x)[:, -1], y).item()\n" + _S_TAIL,
                    en=_S_EN + ": your first loss is measured after learning the batch. The first loss is the one before any training: compute it from `logits`, before `overfit_one_batch`.",
                    nl=_S_NL + ": je eerste loss is gemeten na het leren van de batch. De eerste loss is die vóór elke training: bereken hem uit `logits`, vóór `overfit_one_batch`."),
            Mistake("first-place", wrong=_s(first="    first = F.cross_entropy(logits[:, 0], y).item()\n"),
                    en=_S_EN + ": your first loss uses the logits of the first place. The guess for `c` is at the last place: `logits[:, -1]`.",
                    nl=_S_NL + ": je eerste loss gebruikt de logits van de eerste plek. De gok voor `c` staat op de laatste plek: `logits[:, -1]`."),
            Mistake("shape-vocab", wrong=_s(shape="    shape = logits.shape[-1] == vocab\n"),
                    en=_S_EN + ": your shape check does not look at the number of places. Compare the whole shape, `tuple(logits.shape)`, with `(x.shape[0], x.shape[1], vocab)`: a model that loses a place has the wrong shape too, though its last number is right.",
                    nl=_S_NL + ": je check van de vorm kijkt niet naar het aantal plekken. Vergelijk de hele vorm, `tuple(logits.shape)`, met `(x.shape[0], x.shape[1], vocab)`: een model dat een plek kwijtraakt, heeft ook de verkeerde vorm, al klopt zijn laatste getal."),
            Mistake("other-lr", matches=lambda got, text: re.search(r"overfit_one_batch\([^)]*,\s*(?!1e-3\b|0\.001\b)[\d.e-]+\s*\)", text) is not None,
                    en=_S_EN + ": your batch is learned at another learning rate. The kit's check uses the paper's, `1e-3`, so that it tests the run you will do: a model can learn one batch at one learning rate and not at another.",
                    nl=_S_NL + ": je batch wordt met een andere learning rate geleerd. De check van de kit gebruikt die van de paper, `1e-3`, zodat hij de run test die je gaat doen: een model kan één batch bij de ene learning rate leren en bij een andere niet."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace the `...` with the three lines the comments describe: `shape`, `first` and `losses`.",
                      "nl": "Vervang de `...` door de drie regels die het commentaar beschrijft: `shape`, `first` en `losses`."},
            "wrong": {"en": _S_EN + ". Does `shape` compare the whole shape of `logits`, is `first` the loss at the last place before any training, and do the `losses` come from `overfit_one_batch` with the learning rate `1e-3`?",
                      "nl": _S_NL + ". Vergelijkt `shape` de hele vorm van `logits`, is `first` de loss op de laatste plek vóór elke training, en komen de `losses` uit `overfit_one_batch` met learning rate `1e-3`?"},
            "almost": {"en": "Your `sanity` passes a model that works. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does `shape` compare all three numbers of the shape, and are the two limits of the report, `0.5` and `0.05`, as they were given?",
                       "nl": "Je `sanity` laat een model dat werkt slagen. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Vergelijkt `shape` alle drie de getallen van de vorm, en staan de twee grenzen van het rapport, `0.5` en `0.05`, er nog zoals ze gegeven waren?"},
            "correct": {"en": "Your sanity check tests the shape, the first loss and whether the model can learn one batch, and says what it found. It is now in your project, as `work/projects/repro/sanity.py`.",
                        "nl": "Je sanity check test de vorm, de eerste loss en of het model één batch kan leren, en zegt wat hij vond. Hij staat nu in je project, als `work/projects/repro/sanity.py`."},
            "kept": {"en": "Your sanity check is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je sanity check is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "which-bug" is the choice question in the lesson.
REQUIRED = ["overfit", "best-loss", "which-bug", "sanity"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("overfit", _O_START, "empty", None),
    ("overfit", _O, "correct", None),
    ("overfit", _o(loss=_O_LOSS.replace("[:, -1]", "[:, 0]")), "wrong", "first-place"),
    ("overfit", _o(step="        loss.backward()\n        opt.step()\n"), "wrong", "no-zero"),
    ("overfit", _o(step="        opt.zero_grad()\n        loss.backward()\n"), "wrong", "no-step"),
    ("overfit", _O_HEAD + _O_LOSS + _O_STEP + "        losses.append(F.cross_entropy(model(x)[:, -1], y).item())\n" + _O_TAIL, "wrong", "keep-after"),
    ("overfit", _o(keep="        losses.append(loss)\n"), "error", "keep-tensor"),
    ("overfit", _O.replace(", weight_decay=0.0", ""), "almost", None),
    ("best-loss", _BEST_START, "empty", None),
    ("best-loss", _BEST, "correct", None),
    ("best-loss", _best(add="        total = total + counts[a] * math.log(len(rows))\n"), "wrong", "ln-n"),
    ("best-loss", _best(add="        total = total + math.log(counts[a])\n"), "wrong", "once-per-a"),
    ("best-loss", _best(tail="    return total\n"), "wrong", "no-mean"),
    ("best-loss", _best(count="        counts[c] = counts.get(c, 0) + 1\n"), "wrong", "count-c"),
    ("sanity", _S_START, "empty", None),
    ("sanity", _S, "correct", None),
    ("sanity", _S_HEAD + _S_SHAPE + _S_LOSSES + "    first = F.cross_entropy(model(x)[:, -1], y).item()\n" + _S_TAIL, "wrong", "first-trained"),
    ("sanity", _s(first="    first = F.cross_entropy(logits[:, 0], y).item()\n"), "wrong", "first-place"),
    ("sanity", _s(shape="    shape = logits.shape[-1] == vocab\n"), "wrong", "shape-vocab"),
    ("sanity", _s(losses="    losses = overfit_one_batch(model, x, y, steps, 0.05)\n"), "wrong", "other-lr"),
    ("sanity", _s(shape="    shape = tuple(logits.shape[:2]) == tuple(x.shape)\n"), "almost", None),
    ("sanity", _s(tail=_S_TAIL.replace("abs(first - math.log(vocab))", "first - math.log(vocab)")), "almost", None),
]
