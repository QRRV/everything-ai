# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-rmsnorm: RMSNorm, layer norm without the mean. And the demo: the learner's GPT trained twice
on Tiny Shakespeare, once with its layer norms and once with the learner's RMSNorm in their place."""
import importlib
import re
import time

import numpy as np

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, run_code, with_time_limit

BUILDS_ON = (("llm", "attention", "attn-transformer-block", "causal-self-attention"),
             ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"))
projects.builds_on(*BUILDS_ON)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def norms_picture(labels: dict) -> str:
    """One row of six numbers, x = [3, 5, 4, 6, 5, 7], and what layer norm and RMSNorm make of it, with gamma 1 and
    beta 0. They are values flowing through the network: the input as data, both outputs as hidden values, layer norm
    open and RMSNorm filled (style guide 1.7)."""
    from everything_ai import chart
    x = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 7.0])
    ln = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
    rms = x / np.sqrt((x * x).mean() + 1e-6)
    cols = list(range(1, 7))
    p = chart.Panel((0.5, 8.6), (-2, 7.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=cols, yticks=[-2, 0, 2, 4, 6])
    p.line([0.5, 6.5], [0, 0], "ink-3", 0.8)
    # the two outputs end close together, at 1.55 and 1.36: their names go above and below
    for ys, role, hollow, name, shift in ((x, "data", False, labels["input"], 0), (ln, "hidden", True, labels["ln"], 0.6),
                                          (rms, "hidden", False, labels["rms"], -0.8)):
        p.line(cols, ys, role, 1.4)
        p.points(cols, ys, role, 4, hollow=hollow)
        p.text(6.3, float(ys[-1]) + shift, name, "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=270)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- the demo: layer norm against RMSNorm in the learner's GPT ----------------------------------------------------------------

DATA = "data/tiny-shakespeare.txt"
SIZES = {"d": 64, "heads": 4, "layers": 2, "block": 64}
STEPS, BATCH, LR = 400, 32, 3e-3


def _swap(model, RMSNorm) -> int:
    """Put an RMSNorm of the same width in the place of every nn.LayerNorm in the model; how many it replaced."""
    import torch
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.LayerNorm):
                setattr(parent, name, RMSNorm(child.normalized_shape[-1]))
                count += 1
    return count


def _run(GPT, RMSNorm, ids, n_chars: int, seed: int = 0, show=None, name: str = "") -> dict:
    """Train one small GPT on the first 90% of the ids for STEPS mini-batches, the same ones for every run, and give
    its validation loss on the last 10%, its parameters, its norms swapped and its seconds. `show` hears every 10
    steps the loss of each of them ("fresh"), with the run's `name`, for a live chart."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    s = SIZES
    model = GPT(n_chars, s["d"], s["heads"], s["layers"], s["block"])
    swapped = _swap(model, RMSNorm) if RMSNorm is not None else 0
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    cut = int(0.9 * len(ids))
    train, val = ids[:cut], ids[cut:]
    T = s["block"]
    g = torch.Generator().manual_seed(seed)
    start, fresh = time.perf_counter(), []
    for step in range(STEPS):
        i = torch.randint(0, len(train) - T - 1, (BATCH,), generator=g)
        x = torch.stack([train[j:j + T] for j in i])
        y = torch.stack([train[j + 1:j + T + 1] for j in i])
        loss = F.cross_entropy(model(x).reshape(-1, n_chars), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        fresh.append((step + 1, loss.item()))
        if show is not None and (step % 10 == 9 or step == STEPS - 1):
            show({"run": name, "step": step + 1, "steps": STEPS, "fresh": fresh})
            fresh = []
    seconds = time.perf_counter() - start
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        i = torch.randint(0, len(val) - T - 1, (256,), generator=g)
        x = torch.stack([val[j:j + T] for j in i])
        y = torch.stack([val[j + 1:j + T + 1] for j in i])
        val_loss = float(F.cross_entropy(model(x).reshape(-1, n_chars), y.reshape(-1)))
    return {"val_loss": val_loss, "params": sum(p.numel() for p in model.parameters()), "swapped": swapped, "seconds": seconds}


def demo(code: str, show=None) -> dict | str:
    """The learner's GPT trained twice, with its layer norms and with the learner's RMSNorm in their place; `show`
    hears the steps of both runs, "layer" and then "rms". "norms" when their GPT has no nn.LayerNorm to swap, "error"
    when something stops with an error."""
    import torch
    yours = run_code(code)
    if yours.error is not None or "RMSNorm" not in yours.names:
        return "error"
    RMSNorm = yours.names["RMSNorm"]
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    try:
        GPT = importlib.import_module("llm.gpt").GPT
        book = (paths.ROOT / DATA).read_text(encoding="utf-8")
        chars = sorted(set(book))
        index = {ch: i for i, ch in enumerate(chars)}
        ids = torch.tensor([index[ch] for ch in book])
        s = SIZES   # a GPT without nn.LayerNorm has nothing to swap: say so before a minute of training, and a chart of "your RMSNorm" without it
        if not any(isinstance(m, torch.nn.LayerNorm) for m in GPT(len(chars), s["d"], s["heads"], s["layers"], s["block"]).modules()):
            return "norms"
        runs = with_time_limit(lambda: [_run(GPT, None, ids, len(chars), show=show, name="layer"),
                                        _run(GPT, RMSNorm, ids, len(chars), show=show, name="rms")], 600)
    except Exception:
        return "error"
    finally:
        projects.forget()
    if runs[1]["swapped"] == 0:
        return "norms"
    return {"layer": runs[0], "rms": runs[1], "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}


# --- exercise 1: RMSNorm ------------------------------------------------------------------------------------------------------

_HEAD = "import torch\nfrom torch import nn\n\n\nclass RMSNorm(nn.Module):\n    def __init__(self, d, eps=1e-6):\n        super().__init__()\n"
_INIT = "        self.weight = nn.Parameter(torch.ones(d))\n        self.eps = eps\n"
_FWD = ("        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)\n        return x / rms * self.weight\n")
_START = (_HEAD + "        # one learned scale per feature, starting at 1, and eps\n        ...\n\n    def forward(self, x):\n"
          "        # divide each row by the root of the mean of its squares, then scale every feature\n        ...\n")


def _r(init: str = _INIT, fwd: str = _FWD) -> str:
    return _HEAD + init + "\n    def forward(self, x):\n" + fwd


_SCENARIOS = [
    # two rows of 2 features: [3, 4] and [1, 1]
    "import torch\n\nnorm = RMSNorm(2)\nx = torch.tensor([[3.0, 4.0], [1.0, 1.0]])\nnorm(x).tolist()",
    # the parameters of a norm of 8 features: how many numbers, and their sum at the start
    "norm = RMSNorm(8)\n[sum(p.numel() for p in norm.parameters()), float(norm.weight.sum())]",
    # a mini-batch of 2 texts of 3 tokens with 5 features, with a weight of 1 to 5: the last token of the second text
    "import torch\n\ntorch.manual_seed(0)\nnorm = RMSNorm(5)\nwith torch.no_grad():\n    norm.weight.copy_(torch.arange(1.0, 6.0))\n"
    "x = 3 * torch.randn(2, 3, 5) + 1\nnorm(x)[1, 2].tolist()",
    # a large eps: a row of ones, and a row of zeros
    "import torch\n\nnorm = RMSNorm(3, eps=1.0)\nnorm(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])).tolist()",
]

_SCENARIO = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "rmsnorm": ClassBox(
        name="rmsnorm",
        start=_START,
        solution=_r(),
        scenarios=_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("mean", wrong=_r(fwd="        x = x - x.mean(dim=-1, keepdim=True)\n" + _FWD),
                    en=_SCENARIO + ": you take the mean of the row off first. That is layer normalization without beta; RMSNorm divides the row as it is.",
                    nl=_SCENARIO_NL + ": je trekt eerst het gemiddelde van de rij af. Dat is layer normalization zonder beta; RMSNorm deelt de rij zoals hij is."),
            Mistake("no-root", wrong=_r(fwd="        return x / ((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.weight\n"),
                    en=_SCENARIO + ": you divide by the mean of the squares itself. Take its square root first, `torch.sqrt`, so that the row keeps its own unit.",
                    nl=_SCENARIO_NL + ": je deelt door het gemiddelde van de kwadraten zelf. Neem er eerst de wortel van, `torch.sqrt`, zodat de rij zijn eigen eenheid houdt."),
            Mistake("no-keepdim", wrong=_r(fwd=_FWD.replace(", keepdim=True", "")),
                    en=_SCENARIO + ": without `keepdim=True`, the mean of each row loses its axis, and the division lines up the means of the rows with the features. Keep the axis, so that each row is divided by its own number.",
                    nl=_SCENARIO_NL + ": zonder `keepdim=True` verliest het gemiddelde van elke rij zijn axis, en zet de deling de gemiddelden van de rijen tegenover de features. Houd de axis, zodat elke rij door zijn eigen getal gedeeld wordt."),
            Mistake("wrong-axis", wrong=_r(fwd=_FWD.replace("dim=-1", "dim=0")),
                    en=_SCENARIO + ": you take the mean along the first axis, over the rows. RMSNorm works on each token by itself: the mean of the squares of its own features, along the last axis, `dim=-1`.",
                    nl=_SCENARIO_NL + ": je neemt het gemiddelde langs de eerste axis, over de rijen. RMSNorm werkt op elk token apart: het gemiddelde van de kwadraten van zijn eigen features, langs de laatste axis, `dim=-1`."),
            Mistake("zeros", wrong=_r(init=_INIT.replace("torch.ones(d)", "torch.zeros(d)")),
                    en=_SCENARIO + ": your weight starts at 0, so every output is 0, and the norm wipes out its input until training has grown the weight. Start it at 1, `torch.ones(d)`, so that a new norm only rescales.",
                    nl=_SCENARIO_NL + ": je weight begint bij 0, dus elke output is 0, en de normalisatie veegt zijn input weg tot de training de weight heeft laten groeien. Laat hem bij 1 beginnen, `torch.ones(d)`, zodat een nieuwe normalisatie alleen herschaalt."),
            Mistake("not-parameter", wrong=_r(init=_INIT.replace("nn.Parameter(torch.ones(d))", "torch.ones(d)")), state="almost",
                    en="Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ": your weight is a plain tensor, so `parameters()` does not find it and an optimizer never changes it. Make it a parameter: `nn.Parameter(torch.ones(d))`.",
                    nl="Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ": je weight is een gewone tensor, dus `parameters()` vindt hem niet en een optimizer verandert hem nooit. Maak er een parameter van: `nn.Parameter(torch.ones(d))`."),
            Mistake("no-weight", wrong=_r(fwd=_FWD.replace(" * self.weight", "")), state="almost",
                    en="Right while the weight is 1. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ": multiply by `self.weight`, so that the model can learn a scale per feature.",
                    nl="Goed zolang de weight 1 is. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ": vermenigvuldig met `self.weight`, zodat het model per feature een schaal kan leren."),
            Mistake("eps-outside", wrong=_r(fwd="        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True)) + self.eps\n        return x / rms * self.weight\n"),
                    state="almost",
                    en="Right with a tiny eps. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ": you add eps after the square root. Add it to the mean of the squares, inside `torch.sqrt`, as layer normalization adds it to the variance.",
                    nl="Goed met een piepkleine eps. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ": je telt eps op na de wortel. Tel hem op bij het gemiddelde van de kwadraten, binnen `torch.sqrt`, zoals layer normalization hem bij de variantie optelt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in `__init__` the weight and eps, in `forward` the division and the scale.",
                      "nl": "Vervang de twee `...`: in `__init__` de weight en eps, in `forward` de deling en de schaal."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": _SCENARIO + ". Do you divide each row by the square root of the mean of its squares plus eps, and multiply by a weight that starts at 1?",
                      "nl": _SCENARIO_NL + ". Deel je elke rij door de wortel van het gemiddelde van zijn kwadraten plus eps, en vermenigvuldig je met een weight die bij 1 begint?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your norm work for tokens in a mini-batch, with any weight and any eps?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Werkt je normalisatie voor tokens in een mini-batch, met elke weight en elke eps?"},
            "correct": {"en": "Your RMSNorm divides every token by the root of the mean of its squares and scales each feature: one weight per feature, and no beta.",
                        "nl": "Je RMSNorm deelt elk token door de wortel van het gemiddelde van zijn kwadraten en schaalt elke feature: één weight per feature, en geen beta."},
        },
    ),
}

# The lesson is done when these are correct. "betas" is the choice question, "why-no-mean" the open question.
REQUIRED = ["rmsnorm", "betas", "why-no-mean"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rmsnorm", _START, "empty", None),
    ("rmsnorm", _r(), "correct", None),
    ("rmsnorm", _r(fwd="        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n"), "correct", None),
    ("rmsnorm", _r(fwd=_FWD.replace("(x * x).mean(dim=-1, keepdim=True)", "(x ** 2).sum(dim=-1, keepdim=True) / x.shape[-1]")), "correct", None),
    ("rmsnorm", _r(fwd="        x = x - x.mean(dim=-1, keepdim=True)\n" + _FWD), "wrong", "mean"),
    ("rmsnorm", _r(fwd="        return x / ((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.weight\n"), "wrong", "no-root"),
    ("rmsnorm", _r(fwd=_FWD.replace(", keepdim=True", "")), "wrong", "no-keepdim"),
    ("rmsnorm", _r(fwd=_FWD.replace("dim=-1", "dim=0")), "wrong", "wrong-axis"),
    ("rmsnorm", _r(init=_INIT.replace("torch.ones(d)", "torch.zeros(d)")), "wrong", "zeros"),
    ("rmsnorm", _r(init=_INIT.replace("nn.Parameter(torch.ones(d))", "torch.ones(d)")), "almost", "not-parameter"),
    ("rmsnorm", _r(init=_INIT.replace("nn.Parameter(torch.ones(d))", "torch.ones(d, requires_grad=True)")), "almost", "not-parameter"),
    ("rmsnorm", _r(fwd=_FWD.replace(" * self.weight", "")), "almost", "no-weight"),
    ("rmsnorm", _r(fwd="        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True)) + self.eps\n        return x / rms * self.weight\n"), "almost", "eps-outside"),
]
