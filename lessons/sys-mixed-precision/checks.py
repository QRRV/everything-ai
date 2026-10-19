# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-mixed-precision: train_step, one training step with the forward pass and the loss under
autocast in BF16 and the weights in FP32. And the demo: an MLP for digits trained in FP32 and in mixed precision on the
learner's own machine."""
import re
import time

import numpy as np

from everything_ai import paths
from everything_ai.check import Mistake, ScenarioBox, run_code

# --- exercise 1: train_step ---------------------------------------------------------------------------------------------------

_HEAD = "import torch\nimport torch.nn.functional as F\n\n\ndef train_step(model, opt, x, y, device_type):\n"
_START = (_HEAD + "    opt.zero_grad()\n    # the forward pass and the loss, with the matrix multiplications in BF16\n    ...\n"
          "    # backward and the step, as always\n    loss.backward()\n    opt.step()\n    return loss.item()\n")
_FORWARD = "    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):\n        logits = model(x)\n        loss = F.cross_entropy(logits, y)\n"


def _m(forward: str = _FORWARD, zero: str = "    opt.zero_grad()\n", before: str = "") -> str:
    return _HEAD + before + zero + forward + "    loss.backward()\n    opt.step()\n    return loss.item()\n"


_PURE = _m(before="    model.to(torch.bfloat16)\n", forward="    logits = model(x.to(torch.bfloat16))\n    loss = F.cross_entropy(logits.float(), y)\n")
_PLAIN = _m(forward="    logits = model(x)\n    loss = F.cross_entropy(logits, y)\n")
_FP16 = _m(forward=_FORWARD.replace("bfloat16", "float16"))
_LOSS_OUT = _m(forward="    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):\n        logits = model(x)\n"
                       "    loss = F.cross_entropy(logits, y)\n")

# One nn.Linear is the model: a layer the learner knows, with a forward hook as in nn-activation-stats.
_SETUP = ("import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Linear(4, 3)\n"
          "seen = []  # the format the layer calculates in\nhook = model.register_forward_hook(lambda m, i, o: seen.append(str(o.dtype)))\n")
_SCENARIOS = [
    # the formats: the layer in BF16, the weights in FP32
    _SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n"
             "train_step(model, opt, torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]), \"cpu\")\n"
             "[seen[0], str(model.weight.dtype)]",
    # a small update reaches weights of 1.0: in BF16 it would round away
    _SETUP + "torch.nn.init.ones_(model.weight)  # every weight 1.0, and a small learning rate\n"
             "opt = torch.optim.SGD(model.parameters(), lr=0.001)\n"
             "train_step(model, opt, torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]), \"cpu\")\n"
             "bool((model.weight != 1).any())",
    # the loss it gives back, calculated in FP32 from the BF16 logits; a loss calculated in BF16 differs in the third digit
    _SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n"
             "loss = train_step(model, opt, torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]), \"cpu\")\n"
             "round(float(loss), 4)",
    # two steps in a row, each from zero gradients. Two decimals: the gradients that pile up without zero_grad change
    # the sum by about 0.03 (2.28 against 2.25), a loss calculated in BF16 by less than 0.001
    _SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n"
             "x, y = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])\n"
             "first = train_step(model, opt, x, y, \"cpu\")\nsecond = train_step(model, opt, x, y, \"cpu\")\n"
             "round(model.weight.abs().sum().item(), 2)",
]


def _formats(got) -> list | None:
    """The value of the first scenario, [the format the layer calculates in, the format of the weights], or None."""
    return got if isinstance(got, list) and len(got) == 2 and all(str(v).startswith("torch.") for v in got) else None


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "train-step": ScenarioBox(
        name="train-step",
        start=_START,
        solution=_m(),
        scenarios=_SCENARIOS,
        first=[_DOTS],
        mistakes=[
            Mistake("model-converted", matches=lambda got, text: _formats(got) is not None and got[1] != "torch.float32",
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`: the weights of your model are no longer FP32, so a small update rounds away, as in *Fewer bits*. Leave the model and `x` as they are, and put only the forward pass and the loss under `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):`; autocast makes the BF16 copies itself.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`: de weights van je model zijn geen FP32 meer, dus een kleine update rondt weg, zoals in *Minder bits*. Laat het model en `x` zoals ze zijn, en zet alleen de forward pass en de loss onder `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):`; autocast maakt de BF16-kopieën zelf."),
            Mistake("no-autocast", matches=lambda got, text: _formats(got) == ["torch.float32", "torch.float32"],
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe layer calculates in `torch.float32`: your step is plain FP32. Put the forward pass and the loss under `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\nrekent de layer in `torch.float32`: je stap is gewoon FP32. Zet de forward pass en de loss onder `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):`."),
            Mistake("float16", matches=lambda got, text: _formats(got) == ["torch.float16", "torch.float32"],
                    en="Your layer calculates in FP16: autocast takes `dtype=torch.float16` as well, but then small gradients can become 0 unless you add loss scaling. Use `dtype=torch.bfloat16`, which needs none.",
                    nl="Je layer rekent in FP16: autocast neemt ook `dtype=torch.float16`, maar dan kunnen kleine gradients 0 worden, tenzij je loss scaling toevoegt. Gebruik `dtype=torch.bfloat16`, dat het niet nodig heeft."),
            Mistake("no-zero-grad", wrong=_m(zero=""), state="almost",
                    en="Your second step adds its gradients to those of the first, because nothing sets them back to 0. Keep `opt.zero_grad()` at the start of the step.",
                    nl="Je tweede stap telt zijn gradients op bij die van de eerste, omdat niets ze terugzet naar 0. Houd `opt.zero_grad()` aan het begin van de stap."),
            Mistake("loss-bf16", wrong=_LOSS_OUT, state="almost",
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: your loss has only the 2 or 3 digits of BF16. It is calculated outside the `with` block, from the BF16 logits. Put the loss inside the block too: there autocast calculates it in FP32.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je loss heeft maar de 2 of 3 cijfers van BF16. Hij wordt buiten het `with`-blok uitgerekend, uit de BF16-logits. Zet de loss ook binnen het blok: daar rekent autocast hem in FP32 uit."),
            Mistake("two-formats", error=lambda e: isinstance(e, RuntimeError) and "same dtype" in str(e),
                    en="Python stops: the layer gets its input and its weights in two different formats, BF16 and FP32, and cannot multiply them. Do not convert `x` or the model yourself: inside `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):`, the layer makes BF16 copies of both.",
                    nl="Python stopt: de layer krijgt zijn input en zijn weights in twee verschillende formaten, BF16 en FP32, en kan ze niet vermenigvuldigen. Zet `x` en het model niet zelf om: binnen `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):` maakt de layer van allebei BF16-kopieën."),
        ],
        # Right on the CPU, where the check runs, but not on a GPU: autocast without a dtype picks BF16 on the CPU and
        # FP16 on a GPU, and autocast for the CPU leaves a GPU's calculations in FP32.
        form=[
            Mistake("cpu-only", state="almost",
                    matches=lambda got, text: re.search(r"""autocast\(\s*(?:device_type\s*=\s*)?["']cpu["']|torch\.cpu\.amp""", text) is not None,
                    en="Right here, on the CPU. But your step always asks autocast for the CPU, so on a GPU the layer would calculate in FP32. Pass `device_type` on: `torch.autocast(device_type=device_type, dtype=torch.bfloat16)`.",
                    nl="Goed hier, op de CPU. Maar je stap vraagt autocast altijd voor de CPU, dus op een GPU zou de layer in FP32 rekenen. Geef `device_type` door: `torch.autocast(device_type=device_type, dtype=torch.bfloat16)`."),
            Mistake("no-dtype", state="almost", matches=lambda got, text: "autocast(" in text and "bfloat16" not in text,
                    en="Right here, on the CPU, where autocast picks BF16 by itself. On an NVIDIA GPU it picks FP16, which needs loss scaling. Say what you want: `dtype=torch.bfloat16`.",
                    nl="Goed hier, op de CPU, waar autocast zelf BF16 kiest. Op een NVIDIA-GPU kiest hij FP16, en dat heeft loss scaling nodig. Zeg wat je wilt: `dtype=torch.bfloat16`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the forward pass and the cross-entropy loss, inside a `with torch.autocast(...)` block.",
                      "nl": "Vervang de `...` door de forward pass en de cross-entropy-loss, binnen een `with torch.autocast(...)`-blok."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does the forward pass run under autocast in BF16, while the model stays in FP32?",
                      "nl": "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Draait de forward pass onder autocast in BF16, terwijl het model in FP32 blijft?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does a small update reach the weights, is the loss calculated in FP32, and does each step start from zero gradients?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Komt een kleine update bij de weights aan, wordt de loss in FP32 uitgerekend, en begint elke stap met gradients van nul?"},
            "correct": {"en": "Your step is right in all {n} cases: the layer calculates in BF16, the weights and the loss stay FP32, and even a small update reaches the weights.",
                        "nl": "Je stap klopt in alle {n} gevallen: de layer rekent in BF16, de weights en de loss blijven FP32, en zelfs een kleine update komt bij de weights aan."},
        },
    ),
}

# The lesson is done when these are correct. "what-stays" is the choice question, "scaling" the open question.
REQUIRED = ["train-step", "what-stays", "scaling"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("train-step", _START, "empty", None),
    ("train-step", _m(), "correct", None),
    ("train-step", _m(forward="    with torch.autocast(device_type, torch.bfloat16):\n        loss = F.cross_entropy(model(x), y)\n"), "correct", None),
    ("train-step", _HEAD + "    opt.zero_grad()\n    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):\n        logits = model(x)\n"
     "        loss = F.cross_entropy(logits, y)\n        loss.backward()\n    opt.step()\n    return loss.item()\n", "correct", None),
    ("train-step", _m(forward="    with torch.amp.autocast(device_type, dtype=torch.bfloat16):\n        logits = model(x)\n"
                              "    loss = F.cross_entropy(logits.float(), y)\n"), "correct", None),
    # a GradScaler with BF16 does no harm, and returning the loss tensor counts as a number (lesson template 4.3)
    ("train-step", _HEAD + "    scaler = torch.amp.GradScaler(device_type)\n    opt.zero_grad()\n" + _FORWARD +
     "    scaler.scale(loss).backward()\n    scaler.step(opt)\n    scaler.update()\n    return loss\n", "correct", None),
    ("train-step", _PURE, "wrong", "model-converted"),
    ("train-step", _m(before="    model.half()\n", forward=_FORWARD.replace("bfloat16", "float16")), "wrong", "model-converted"),
    ("train-step", _PLAIN, "wrong", "no-autocast"),
    ("train-step", _FP16, "wrong", "float16"),
    ("train-step", _m(zero=""), "almost", "no-zero-grad"),
    ("train-step", _LOSS_OUT, "almost", "loss-bf16"),
    ("train-step", _m(forward="    logits = model(x.to(torch.bfloat16))\n    loss = F.cross_entropy(logits, y)\n"), "error", "two-formats"),
    ("train-step", _m(forward=_FORWARD.replace("device_type=device_type", "\"cpu\"")), "almost", "cpu-only"),
    ("train-step", _m(forward=_FORWARD.replace(", dtype=torch.bfloat16", "")), "almost", "no-dtype"),
    ("train-step", _START.replace("    ...\n", _FORWARD), "correct", None),
]


def train_step_of(text: str | None):
    """The learner's train_step from the code of their box, when it is right, else the course's: (function, theirs)."""
    if text:
        ran = run_code(text)
        fn = ran.names.get("train_step") if ran.error is None else None
        if callable(fn):
            return fn, True
    return run_code(_m()).names["train_step"], False


# --- the demo: digits in FP32 and in mixed precision --------------------------------------------------------------------------

def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(where: str):
    import torch
    if where == "cuda":
        torch.cuda.synchronize()
    elif where == "mps":
        torch.mps.synchronize()


def _bf16_runs(where: str) -> bool:
    """Whether a matrix multiplication under autocast in BF16 really gives BF16 on this device. Not on a Mac before
    macOS 14, for one: there the GPU has no BF16, and autocast stops with an error or turns itself off."""
    import warnings

    import torch
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = torch.ones(2, 2, device=where)
            with torch.autocast(device_type=where, dtype=torch.bfloat16):
                out = a @ a
            _sync(where)
        return out.dtype == torch.bfloat16
    except Exception:   # any error of the device itself: the demo then runs on the CPU
        return False


def _name(dtype) -> str:
    import torch
    return {torch.float32: "FP32", torch.bfloat16: "BF16", torch.float16: "FP16"}.get(dtype, str(dtype))


def compare(step, epochs: int = 3, show=None) -> dict:
    """An MLP of 784, 1024, 1024 and 10 trained on the course's 6,000 digits for `epochs` epochs, twice from the same
    start and with the same order of mini-batches: once with a plain FP32 step, once with `step`. Per run the seconds per
    epoch (without the first, which also warms up), the accuracy on the 1,000 test digits, the format the first layer
    calculates in (seen by a forward hook on the first mini-batch) and the format of the weights after training.
    A GPU that cannot calculate in BF16 under autocast is left alone: then both runs are on the CPU, and "moved" says so."""
    import torch
    import torch.nn.functional as F
    where = _device()
    moved = where != "cpu" and not _bf16_runs(where)
    if moved:
        where = "cpu"
    data = np.load(paths.ROOT / "data" / "qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255.0
    Y = torch.tensor(data["train_labels"], dtype=torch.long)
    Xt = torch.tensor(data["test_images"], dtype=torch.float32) / 255.0
    Yt = torch.tensor(data["test_labels"], dtype=torch.long)
    X, Y, Xt, Yt = X.to(where), Y.to(where), Xt.to(where), Yt.to(where)

    def plain(model, opt, x, y, device_type):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.item()

    out = {"device": where, "moved": moved}
    for name, fn in (("fp32", plain), ("mixed", step)):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(784, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
                                    torch.nn.Linear(1024, 10)).to(where)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        seen = []
        hook = model[0].register_forward_hook(lambda m, i, o: seen.append(o.dtype))
        times = []
        for epoch in range(epochs):
            if show:
                show(name, epoch, epochs)
            g = torch.Generator().manual_seed(epoch)
            order = torch.randperm(len(X), generator=g).to(where)
            _sync(where)
            t = time.perf_counter()
            for i in range(0, len(X), 128):
                idx = order[i:i + 128]
                fn(model, opt, X[idx], Y[idx], where)
                if hook is not None:   # one mini-batch is enough to see the format; the first epoch is not timed
                    hook.remove()
                    hook = None
            _sync(where)
            times.append(time.perf_counter() - t)
        with torch.no_grad():
            acc = (model(Xt).argmax(1) == Yt).float().mean().item()
        out[name] = {"seconds": float(np.mean(times[1:])) if len(times) > 1 else times[0], "accuracy": acc,
                     "calculates": _name(seen[0]) if seen else "?", "weights": _name(next(model.parameters()).dtype)}
    return out


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def flow_picture(labels: dict) -> str:
    """One training step in mixed precision, top to bottom: the weights in FP32 are what the model learns, so orange;
    the loss is purple; the rest is ink. Each box names its format. HTML boxes, so the text wraps in a narrow window."""
    import html

    def box(text_, role="ink"):
        line = {"ink": "var(--ea-ink-3)", "param": "var(--ea-param)", "loss": "var(--ea-loss)"}[role]
        ink = {"ink": "var(--ea-ink)", "param": "var(--ea-param-text)", "loss": "var(--ea-loss-text)"}[role]
        return (f'<span style="display:inline-block;padding:0.3rem 0.6rem;border:1.2px solid {line};border-radius:6px;'
                f'background:var(--ea-inset);color:{ink}">{html.escape(text_)}</span>')
    arrow = '<div style="text-align:center;color:var(--ea-ink-3);margin:0.15rem 0">↓</div>'
    rows = [box(labels["weights"], "param"), box(labels["copy"]), box(labels["forward"]), box(labels["loss"], "loss"),
            box(labels["backward"]), box(labels["grads"]), box(labels["update"], "param")]
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:560px;font-family:var(--ea-font-ui);font-size:0.875rem;line-height:1.5">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            + arrow.join(f'<div style="text-align:center">{r}</div>' for r in rows) + '</figure>')
