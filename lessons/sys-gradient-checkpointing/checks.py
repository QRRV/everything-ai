# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-gradient-checkpointing: checkpointed, a stack of blocks run in segments with
torch.utils.checkpoint, measured with saved_tensors_hooks; and best_k, the segment length with the least activation
memory. And the picture: the activation memory of 16 layers for every segment length, kept and recomputed."""
import math
import re
import time

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code

LAYERS = 16


def grouped(n: float, lang: str) -> str:
    return f"{n:,.0f}".replace(",", "." if lang == "nl" else ",")


def units(n: int, k: int) -> tuple:
    """Activation memory in units of one layer's activations, for n layers in segments of k: (kept from the forward
    pass, the inputs of the segments; recomputed during the backward pass, one segment at a time)."""
    return math.ceil(n / k), min(k, n)


def segments_picture(labels: dict) -> str:
    """For 16 layers and segment lengths 1, 2, 4, 8 and 16, the activation memory as a stacked bar: the inputs of the
    segments that are kept (hidden, filled) and the one segment that is recomputed (hidden, light); a line at 17, all
    activations kept without checkpointing: the network's input and the output of every layer. The names of the bars
    stand under them inside the axes, because a chart clips what lies outside."""
    from everything_ai import chart
    ks = [1, 2, 4, 8, 16]
    p = chart.Panel((-0.6, len(ks) - 0.4), (-2.6, 20.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[], yticks=[0, 4, 8, 12, 16])
    for i, k in enumerate(ks):
        kept, seg = units(LAYERS, k)
        for _ in range(3):
            p.rect(i - 0.3, 0, i + 0.3, kept, "hidden")
        p.rect(i - 0.3, kept, i + 0.3, kept + seg, "hidden")
        p.text(i, -1.4, f"k = {k}", "ink-2")
        p.text(i, kept + seg + 1.1, str(kept + seg), "ink")
    p.line([-0.6, len(ks) - 0.4], [LAYERS + 1, LAYERS + 1], "ink-3", 1.2, dashed=True)
    p.text(2, LAYERS + 2.2, labels["plain"], "ink-2")
    return chart.figure([p], alt=labels["alt"], width=330, height=270)


# --- exercise 1: checkpointed ----------------------------------------------------------------------------------------------------

_C_HEAD = ("import torch\nfrom torch.autograd.graph import saved_tensors_hooks\nfrom torch.utils.checkpoint import checkpoint\n\n\n"
           "def kept_bytes(run, blocks):\n"
           "    \"\"\"The bytes the forward pass of run() keeps for the backward pass, parameters left out.\"\"\"\n"
           "    places = {p.data_ptr() for b in blocks for p in b.parameters()}\n    kept = {}\n\n"
           "    def pack(t):\n        if t.data_ptr() not in places:\n            kept[t.data_ptr()] = t.numel() * t.element_size()\n"
           "        return t\n\n"
           "    with saved_tensors_hooks(pack, lambda t: t):\n        out = run()\n    out.sum().backward()\n"
           "    return sum(kept.values())\n\n\n"
           "def run_blocks(blocks, x):\n    for block in blocks:\n        x = torch.relu(block(x))\n    return x\n\n\n"
           "def checkpointed(blocks, x, k):\n")
_C_BODY = ("    for start in range(0, len(blocks), k):\n"
           "        x = checkpoint(run_blocks, blocks[start:start + k], x, use_reentrant=False)\n"
           "    return x\n")
_C_START = (_C_HEAD + "    # run the blocks in segments of k, each segment through checkpoint, with use_reentrant=False\n"
            "    ...\n")


def _c(body: str = _C_BODY) -> str:
    return _C_HEAD + body


_SETUP = ("import torch\n\ntorch.manual_seed(0)\nblocks = [torch.nn.Linear(32, 32) for _ in range(8)]\n"
          "x = torch.randn(16, 32)\n")
_C_SCENARIOS = [
    # in segments of 2: the same output as without checkpoints, and 4 segment inputs kept, of 16 x 32 floats
    _SETUP + "out = checkpointed(blocks, x, 2)\nround((out - run_blocks(blocks, x)).abs().max().item(), 6)",
    _SETUP + "kept_bytes(lambda: checkpointed(blocks, x, 2), blocks)",
    # segments of 3 for 8 blocks: the last one is shorter
    _SETUP + "kept_bytes(lambda: checkpointed(blocks, x, 3), blocks)",
    # the backward pass reaches the first block: the gradient of its weights, or None when it has none
    _SETUP + "checkpointed(blocks, x, 3).sum().backward()\ngrad = blocks[0].weight.grad\nNone if grad is None else round(grad.abs().sum().item(), 4)",
]

_C_WRONG = {
    "no-checkpoint": _c("    return run_blocks(blocks, x)\n"),
    "every-block": _c("    for block in blocks:\n        x = checkpoint(run_blocks, [block], x, use_reentrant=False)\n    return x\n"),
}

# --- exercise 2: best_k ----------------------------------------------------------------------------------------------------------

_B_HEAD = "import math\n\n\ndef best_k(n):\n"
_B_BODY = ("    best = None\n    for k in range(1, n + 1):\n        peak = math.ceil(n / k) + k\n"
           "        if best is None or peak < best_peak:\n            best, best_peak = k, peak\n    return best\n")
_B_START = (_B_HEAD + "    # for every segment length k from 1 to n: the kept segment inputs, ceil(n / k), plus the k recomputed;\n"
            "    # give back the k with the smallest sum, the smallest k when there is a tie\n    ...\n")


def _bk(body: str = _B_BODY) -> str:
    return _B_HEAD + body


_B_WRONG = {
    "no-ceil": _bk(_B_BODY.replace("math.ceil(n / k)", "n / k")),
    "peak": _bk(_B_BODY.replace("    return best\n", "    return best_peak\n")),
    "last-tie": _bk(_B_BODY.replace("peak < best_peak", "peak <= best_peak")),
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your code:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _same(got, want) -> bool:
    try:
        return not isinstance(got, (str, bytes, bool)) and math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "checkpointed": ScenarioBox(
        name="checkpointed",
        start=_C_START,
        solution=_c(),
        scenarios=_C_SCENARIOS,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            # use_reentrant=True, or left out (PyTorch 2.14 then still checkpoints the old way): the segments of x, which
            # needs no gradient, give an output without one, and backward() in kept_bytes stops
            Mistake("reentrant", error=lambda e: "does not require grad" in str(e) or "use_reentrant" in str(e),
                    en="Does every `checkpoint` have `use_reentrant=False`? Without it, or with `use_reentrant=True`, PyTorch checkpoints the older way: that gives no gradients when the input of a segment needs none, as `x` here, and then `backward()` has nothing to go back through.",
                    nl="Krijgt elke aanroep van `checkpoint` `use_reentrant=False` mee? Zonder, of met `use_reentrant=True`, werkt `checkpoint` op de oudere manier: die geeft geen gradients als de input van een segment er geen nodig heeft, zoals `x` hier, en dan kan `backward()` nergens terug doorheen."),
            Mistake("no-gradient", matches=lambda got, text: got is None,
                    en=_SCENARIO + ": after `backward()`, the first block has no gradient. The output is right, but the gradient does not flow back through your segments. Does each segment get the output of the one before as it is, without `.detach()` or `torch.no_grad()`?",
                    nl=_SCENARIO_NL + ": na `backward()` heeft het eerste block geen gradient. De output klopt, maar de gradient stroomt niet terug door je segmenten. Krijgt elk segment de output van het vorige zoals die is, zonder `.detach()` of `torch.no_grad()`?"),
            Mistake("no-checkpoint", wrong=_C_WRONG["no-checkpoint"],
                    en=_SCENARIO + ": the right output, but the forward pass keeps every activation, as without checkpoints. Run each segment of `k` blocks through `checkpoint(run_blocks, blocks[start:start + k], x, use_reentrant=False)`.",
                    nl=_SCENARIO_NL + ": de goede output, maar de forward pass bewaart elke activation, zoals zonder checkpoints. Stuur elk segment van `k` blocks door `checkpoint(run_blocks, blocks[start:start + k], x, use_reentrant=False)`."),
            Mistake("every-block", wrong=_C_WRONG["every-block"],
                    en=_SCENARIO + ": you put a checkpoint around every single block, so the forward pass keeps the input of every block, whatever `k` is. A segment holds `k` blocks: `blocks[start:start + k]`, with `start` stepping by `k`.",
                    nl=_SCENARIO_NL + ": je zet een checkpoint om elk afzonderlijk block, dus de forward pass bewaart de input van elk block, wat `k` ook is. Een segment bevat `k` blocks: `blocks[start:start + k]`, met `start` in stappen van `k`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: a loop over the segments of `k` blocks, each run through `checkpoint`, and the output at the end.",
                      "nl": "Vervang de `...`: een loop over de segmenten van `k` blocks, elk door `checkpoint`, en de output aan het eind."},
            "wrong": {"en": _SCENARIO + ". A segment is `k` blocks in a row, run through `checkpoint` with `run_blocks`: it keeps only the input of the segment. Does every block run once, in order?",
                      "nl": _SCENARIO_NL + ". Een segment is `k` blocks achter elkaar, door `checkpoint` gestuurd met `run_blocks`: het bewaart alleen de input van het segment. Draait elk block één keer, op volgorde?"},
            "almost": {"en": "The output is right. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ": one segment input for every `k` blocks, a shorter last segment included.",
                       "nl": "De output klopt. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ": één input voor elk segment van `k` blocks, ook voor een korter laatste segment."},
            "correct": {"en": "Your checkpointed run gives the same output, and keeps only the input of every segment: {n} scenarios right.",
                        "nl": "Je run met checkpoints geeft dezelfde output, en bewaart alleen de input van elk segment: {n} scenario's goed."},
        },
    ),
    "best-k": FunctionBox(
        name="best-k",
        function="best_k",
        start=_B_START,
        solution=_bk(),
        visible=[(16,)],
        hidden=[(10,), (1,), (30,), (7,)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-ceil", wrong=_B_WRONG["no-ceil"],
                    # also √n rounded, round(math.sqrt(n)): for every n the same k as n / k + k without rounding up
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the best length for `n / k + k` without rounding up, and also √n rounded. But each segment keeps a whole input, also a shorter last one: count `math.ceil(n / k)` inputs, and keep the smallest `k` on a tie.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de beste lengte voor `n / k + k` zonder naar boven af te ronden, en ook √n afgerond. Maar elk segment bewaart een hele input, ook een korter laatste: tel `math.ceil(n / k)` inputs, en houd bij gelijke sommen de kleinste `k`."),
            Mistake("peak", wrong=_B_WRONG["peak"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you give back the smallest memory, and the question is which segment length gives it. Give back `k`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je geeft het kleinste geheugen terug, en de vraag is welke lengte van een segment dat geeft. Geef `k` terug."),
            Mistake("last-tie", wrong=_B_WRONG["last-tie"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: when two lengths need the same memory, you keep the last one. Keep the first, the smallest `k`: replace only on a strictly smaller sum, with `<`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: als twee lengtes evenveel geheugen nodig hebben, houd je de laatste. Houd de eerste, de kleinste `k`: vervang alleen bij een echt kleinere som, met `<`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: for every `k` from 1 to `n`, the kept inputs plus the recomputed segment; keep the `k` with the smallest sum.",
                      "nl": "Vervang de `...`: voor elke `k` van 1 tot en met `n` de bewaarde inputs plus het opnieuw berekende segment; houd de `k` met de kleinste som."},
            "arguments": {"en": "The check calls `{call}`, with the number of layers, and the `def` line of your function has another number of names between its parentheses. Keep `(n)`.",
                          "nl": "De check roept `{call}` aan, met het aantal layers, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For every `k`: `math.ceil(n / k) + k`; give back the `k` where that is smallest, the smallest `k` on a tie.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Voor elke `k`: `math.ceil(n / k) + k`; geef de `k` terug waar dat het kleinst is, bij gelijke sommen de kleinste `k`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for 1 layer, and when `k` does not divide `n`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het ook voor 1 layer, en als `n` niet deelbaar is door `k`?"},
            "correct": {"en": "Your `best_k` is right in all {n} cases: about the square root of `n`, as the formula says.",
                        "nl": "Je `best_k` klopt in alle {n} gevallen: ongeveer de wortel uit `n`, zoals de formule zegt."},
        },
    ),
}

# The lesson is done when these are correct. "same-compute" is the choice question.
REQUIRED = ["checkpointed", "best-k", "same-compute"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("checkpointed", _C_START, "empty", None),
    ("checkpointed", _c(), "correct", None),
    ("checkpointed", _c("    segments = [blocks[i:i + k] for i in range(0, len(blocks), k)]\n    for segment in segments:\n"
                        "        x = checkpoint(run_blocks, segment, x, use_reentrant=False)\n    return x\n"), "correct", None),
] + [("checkpointed", code, "wrong", key) for key, code in _C_WRONG.items()] + [
    ("checkpointed", _c(_C_BODY.replace("use_reentrant=False", "use_reentrant=True")), "error", "reentrant"),
    ("checkpointed", _c(_C_BODY.replace(", use_reentrant=False", "")), "error", "reentrant"),
    ("checkpointed", _c(_C_BODY.replace("run_blocks, blocks[start:start + k], x,", "run_blocks, blocks[start:start + k], x.detach(),")), "wrong", "no-gradient"),
    ("best-k", _B_START, "empty", None),
    ("best-k", _bk(), "correct", None),
    ("best-k", _bk("    return min(range(1, n + 1), key=lambda k: math.ceil(n / k) + k)\n"), "correct", None),
] + [("best-k", code, "wrong", key) for key, code in _B_WRONG.items()] + [
    ("best-k", _bk("    return round(math.sqrt(n))\n"), "wrong", "no-ceil"),
]


# --- the demo: the learner's checkpointed on 16 layers, measured and timed -----------------------------------------------------

def demo(text, width: int = 512, batch: int = 256, runs: int = 5) -> dict:
    """The learner's checkpointed (or the course's) on 16 linear layers, each with a ReLU after it, for every segment length:
    the bytes kept between the passes, and the time of a forward and backward pass, against a run without checkpoints."""
    ran = run_code(text) if text else None
    yours = ran is not None and ran.error is None and callable(ran.names.get("checkpointed"))
    names = ran.names if yours else run_code(_c()).names
    checkpointed, kept_bytes, run_blocks = names["checkpointed"], names["kept_bytes"], names["run_blocks"]
    torch.manual_seed(0)
    blocks = [torch.nn.Linear(width, width) for _ in range(LAYERS)]
    x = torch.randn(batch, width)

    def timed(fn):
        fn().sum().backward()
        best = float("inf")
        for _ in range(runs):   # the fastest of a few runs: the least disturbed by other work on the computer
            start = time.perf_counter()
            fn().sum().backward()
            best = min(best, time.perf_counter() - start)
        return best

    rows = [("-", kept_bytes(lambda: run_blocks(blocks, x), blocks), timed(lambda: run_blocks(blocks, x)))]
    for k in (1, 2, 4, 8, 16):
        rows.append((k, kept_bytes(lambda: checkpointed(blocks, x, k), blocks), timed(lambda: checkpointed(blocks, x, k))))
    return {"rows": rows, "yours": yours, "layer": batch * width * 4}
