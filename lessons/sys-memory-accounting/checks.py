# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-memory-accounting: training_gb, the memory of the weights, the gradients, the optimizer
state and the activations of a training run. And the demo: the activations that the learner's GPT keeps for the
backward pass, measured for growing batches, next to its weights, gradients and AdamW state."""
import importlib
import math
import re

from everything_ai import projects
from everything_ai.check import FunctionBox, Mistake, with_time_limit

projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"))

# Bytes per parameter in FP32: the weight, its gradient, and the optimizer's state.
STATE_BYTES = {"sgd": 0, "momentum": 4, "adamw": 8}


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the demo: what the learner's GPT keeps for the backward pass -------------------------------------------------------------

SIZES = {"vocab": 65, "d": 64, "heads": 4, "layers": 2, "block": 64}   # the small GPT of res-hypothesis-control
BATCHES = [1, 4, 16, 64]


def saved_bytes(model, x) -> int:
    """The bytes of the tensors that the forward pass of `model` on `x` keeps for the backward pass, each storage
    counted once, the parameters left out."""
    from torch.autograd.graph import saved_tensors_hooks
    params = {p.data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        if t.data_ptr() not in params:
            seen[t.data_ptr()] = max(seen.get(t.data_ptr(), 0), t.numel() * t.element_size())
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        loss = model(x).float().sum()
    del loss
    return sum(seen.values())


def measure() -> dict | str:
    """The learner's GPT, or the course's where theirs is missing, with the sizes of SIZES: its parameters, and the
    bytes it keeps for the backward pass for each batch of BATCHES windows of `block` tokens. "error" when their GPT
    stops with an error at these sizes."""
    import torch
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()

    def work():
        GPT = importlib.import_module("llm.gpt").GPT
        torch.manual_seed(0)
        s = SIZES
        model = GPT(s["vocab"], s["d"], s["heads"], s["layers"], s["block"])
        rows = [(b, saved_bytes(model, torch.randint(0, s["vocab"], (b, s["block"])))) for b in BATCHES]
        return sum(p.numel() for p in model.parameters()), rows

    try:
        n, rows = with_time_limit(work, 60)
    except Exception:
        return "error"
    finally:
        projects.forget()
    return {"params": n, "rows": rows, "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}


def bars_picture(labels: dict) -> str:
    """Stacked bars of the memory of using a 7-billion-parameter model in BF16, and of fine-tuning all of it with
    AdamW, against the memory of two GPUs. The GB of each part inside its bar, the names of the parts next to the
    training bar. Weights are what a model learns, so orange; gradients in their own role; the optimizer state in ink."""
    from everything_ai import chart
    n = 7e9
    use = [("param", 2 * n / 1e9)]
    train = [("param", 4 * n / 1e9), ("grad", 4 * n / 1e9), ("ink-2", 8 * n / 1e9)]
    p = chart.Panel((-0.2, 3.3), (-14, 125), title=labels["title"], xlabel="", ylabel=labels["y"], xticks=[],
                    yticks=[0, 25, 50, 75, 100, 125])
    for gb, name, at in ((24, labels["gpu24"], 29), (80, labels["gpu80"], 85)):
        p.line([-0.2, 3.3], [gb, gb], "ink-3", 1.2, dashed=True)
        p.text(3.25, at, name, "ink-3", "end")
    for i, parts in enumerate((use, train)):
        x0 = i * 0.95
        bottom = 0
        for role, gb in parts:
            p.rect(x0, bottom, x0 + 0.7, bottom + gb, role)
            p.text(x0 + 0.35, bottom + gb / 2, f"{gb:g}", "ink", "middle")
            bottom += gb
        p.text(x0 + 0.35, -7, labels["bars"][i], "ink", "middle")
    for y, name in ((14, labels["weights"]), (42, labels["grads"]), (98, labels["state"])):
        p.text(1.75, y, name, "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=300)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: training_gb --------------------------------------------------------------------------------------------------

_HEAD = "def training_gb(n_params, optimizer, activation_gb):\n"
_START = (_HEAD + "    # bytes per parameter of the optimizer's state, in FP32\n    state = {\"sgd\": 0, \"momentum\": 4, \"adamw\": 8}[optimizer]\n"
          "    # the weights, their gradients, the optimizer's state and the activations, in GB of 10**9 bytes\n    ...\n")


def _t(per: str = "4 + 4 + state", scale: str = "1e9", extra: str = " + activation_gb") -> str:
    return _HEAD + ("    state = {\"sgd\": 0, \"momentum\": 4, \"adamw\": 8}[optimizer]\n"
                    f"    return n_params * ({per}) / {scale}{extra}\n")


def _close(got, want) -> bool:
    """A number close to the right one; 1e-6, so that a float32 from NumPy or PyTorch counts too. Text such as "112.0" is
    no number."""
    if isinstance(got, (str, bytes)):
        return False
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _show(v) -> str:
    """A number with at most 4 decimals and never in e-notation: 112000000000, not 1.12e+11. Text as code, in quotes."""
    if isinstance(v, str):
        return '"' + v + '"'
    try:
        return f"{float(v):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return repr(v)


class _Num(float):
    """A number that a check message writes the way the lesson does, such as `7e9`."""

    def __new__(cls, value, shown):
        obj = super().__new__(cls, value)
        obj.shown = shown
        return obj

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return float(self)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "training-gb": FunctionBox(
        name="training-gb",
        function="training_gb",
        start=_START,
        solution=_t(),
        visible=[(_Num(7e9, "7e9"), "adamw", 0)],
        hidden=[(_Num(10.8e6, "10.8e6"), "adamw", 0.5), (_Num(1.5e9, "1.5e9"), "sgd", 3), (_Num(125e6, "125e6"), "momentum", 2.0),
                (_Num(70e9, "70e9"), "adamw", 40)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-grads", wrong=_t(per="4 + state"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is 4 bytes per parameter too few. Do you leave out the gradients, or count the weights and gradients as 2 bytes each? Both take 4 bytes, also in mixed precision, where the master weights and their gradients stay FP32.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is 4 bytes per parameter te weinig. Laat je de gradients weg, of tel je de weights en gradients als 2 bytes per stuk? Ze nemen elk 4 bytes, ook in mixed precision, waar de master weights en hun gradients FP32 blijven."),
            Mistake("no-state", wrong=_t(per="4 + 4"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you leave out the optimizer's state. AdamW keeps two averages per weight, 8 bytes; use `state`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laat de optimizer state weg. AdamW houdt twee gemiddelden per weight bij, 8 bytes; gebruik `state`."),
            Mistake("bytes", wrong=_t(scale="1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in bytes. Divide the bytes by `1e9` to get GB, and add `activation_gb` after that.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in bytes. Deel de bytes door `1e9` om GB te krijgen, en tel daarna `activation_gb` op."),
            Mistake("gib", wrong=_t(scale="2**30"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by `2**30`, a GiB. In this course a GB is `10**9` bytes.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door `2**30`, een GiB. In deze course is een GB `10**9` bytes."),
            Mistake("no-activations", wrong=_t(extra=""), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: the activations are missing. Add `activation_gb` after converting to GB, as it is in GB already.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de activations ontbreken. Tel `activation_gb` op na het omrekenen naar GB, want die is al in GB."),
            Mistake("text", matches=lambda got, text: isinstance(got, str),
                    en="`{call}` gives the text `{got}`, and it should give the number `{want}`. Give back the number itself, without turning it into text.",
                    nl="`{call}` geeft de tekst `{got}`, en dat moet het getal `{want}` zijn. Geef het getal zelf terug, zonder er tekst van te maken."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `return` of the total in GB: the bytes per parameter times `n_params`, divided by `1e9`, plus `activation_gb`.",
                      "nl": "Vervang de `...` door een `return` van het totaal in GB: de bytes per parameter maal `n_params`, gedeeld door `1e9`, plus `activation_gb`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the number of parameters, the optimizer and the activations in GB, and the `def` line of your function has another number of names between its parentheses. Keep `(n_params, optimizer, activation_gb)`.",
                          "nl": "De check roept `{call}` aan, met het aantal parameters, de optimizer en de activations in GB, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n_params, optimizer, activation_gb)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you count 4 bytes for each weight, 4 for its gradient and `state` for the optimizer, then divide by `1e9` and add the activations?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je 4 bytes voor elke weight, 4 voor zijn gradient en `state` voor de optimizer, en deel je daarna door `1e9` en tel je de activations op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every optimizer, and with activations that are not 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het voor elke optimizer, en met activations die niet 0 zijn?"},
            "correct": {"en": "Your function is right in all {n} cases: 16 bytes per parameter with AdamW, 8 with plain SGD, plus the activations.",
                        "nl": "Je functie klopt in alle {n} gevallen: 16 bytes per parameter met AdamW, 8 met gewone SGD, plus de activations."},
        },
    ),
}

# The lesson is done when these are correct. "fine-tune" is the choice question, "out-of-memory" the open question.
REQUIRED = ["training-gb", "fine-tune", "out-of-memory"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("training-gb", _START, "empty", None),
    ("training-gb", _t(), "correct", None),
    ("training-gb", _HEAD + "    state = {\"sgd\": 0, \"momentum\": 4, \"adamw\": 8}[optimizer]\n    weights = 4 * n_params\n    grads = 4 * n_params\n"
     "    opt = state * n_params\n    return (weights + grads + opt) / 10**9 + activation_gb\n", "correct", None),
    ("training-gb", _t(per="4 + state"), "wrong", "no-grads"),
    ("training-gb", _t(per="4 + 4"), "wrong", "no-state"),
    ("training-gb", _t(scale="1"), "wrong", "bytes"),
    ("training-gb", _t(scale="2**30"), "wrong", "gib"),
    ("training-gb", _t(extra=""), "almost", "no-activations"),
    # the activations added to the bytes, before converting: they count for almost nothing
    ("training-gb", _HEAD + "    state = {\"sgd\": 0, \"momentum\": 4, \"adamw\": 8}[optimizer]\n"
     "    return (n_params * (4 + 4 + state) + activation_gb) / 1e9\n", "almost", "no-activations"),
    # weights and gradients as BF16, 2 bytes each: the same numbers as leaving out the gradients
    ("training-gb", _t(per="2 + 2 + state"), "wrong", "no-grads"),
    ("training-gb", _t(extra=" + activation_gb", scale="1e9").replace("return ", "return str(", 1).rstrip("\n") + ")\n", "wrong", "text"),
    # a float32 from NumPy is right too
    ("training-gb", "import numpy as np\n" + _t().replace("return ", "return np.float32(", 1).rstrip("\n") + ")\n", "correct", None),
]
