# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-quantize: quantize, the weights of a matrix as a model computes with them after storing
each in a few bits, with a scale per group of weights; and a choice. The picture shows one group of the course's model
at every number of bits; the demo stores the learner's model of llm-pretrain-capstone (or the course's) in fewer bits
with their own quantize, and measures its size and its loss on stories it never trained on."""
import copy
import importlib.util
import json
import math

from everything_ai import paths, projects
from everything_ai.check import Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _checks("llm-pretrain-capstone", "everything_ai_checks_llm_pretrain_capstone_for_quantize")
B = P.B
BUILDS_ON = P.BUILDS_ON
projects.builds_on(*BUILDS_ON)

STORIES = 400      # the last stories of data/tinystories-small.txt: in the validation shard, never trained on
TOKENS = 20_000    # the ids of them every model is scored on: enough for a loss to two decimals, in seconds on a CPU
ENCODED = 100      # the first stories of those that are encoded: about 27,000 ids, the same first TOKENS as of all
SCALE_BITS = 16    # each group's scale is stored in 16 bits
GROUP = 32         # the group size of the demo, as many 4-bit formats use
SETTINGS = [(32, None), (8, GROUP), (4, GROUP), (4, "row"), (3, GROUP), (2, GROUP)]   # (bits, group) of the demo
PICTURE = ("blocks.0.mlp.down.weight", 0)   # the picture's group: the first GROUP weights of this row of the course's model


def number(x: float, digits: int, lang: str) -> str:
    return P.number(x, digits, lang)


def qmax(bits: int) -> int:
    """The largest whole number q of a signed format of `bits` bits that has as many levels below 0 as above."""
    return 2 ** (bits - 1) - 1


def course_quantize(w, bits: int, group: int):
    """The course's quantize, as the solution of the exercise writes it."""
    return run_code(Q_CODE).names["quantize"](w, bits, group)


def text() -> str:
    """The stories every model is scored on: the last STORIES of data/tinystories-small.txt, joined by <|endoftext|>.
    They lie in the validation shard of llm-data-pipeline, so the model of llm-pretrain-capstone never trained on them."""
    stories = [s.strip() for s in (paths.ROOT / "data" / "tinystories-small.txt").read_text(encoding="utf-8").split(B.END)
               if s.strip()]
    return B.END.join(stories[-STORIES:])


def model_and_ids(source: str | None = None) -> tuple:
    """The model of llm-pretrain-capstone (the learner's, or the course's; "course" asks for the course's), its
    information, where it comes from, and the first TOKENS + 1 ids of the stories."""
    model, info, where = P.load(source)
    tools = run_code("from llm.encode import encode\nfrom llm.decode import decode").names
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    ids = tools["encode"](B.END.join(text().split(B.END)[:ENCODED]), merges)[:TOKENS + 1]
    return model, info, where, ids


def val_loss(model, ids) -> float:
    """The mean loss per token of the ids, in windows of the model's block that follow each other."""
    import torch
    import torch.nn.functional as F
    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids) - 1, P.BLOCK):
            window = ids[start:start + P.BLOCK + 1]
            logits = model(torch.tensor([window[:-1]]))[0]
            total += F.cross_entropy(logits, torch.tensor(window[1:]), reduction="sum").item()
            n += len(window) - 1
    return total / n


def matrices(model) -> list[str]:
    """The names of the model's weight matrices: those of its layers and of its embedding, which quantization stores in
    fewer bits. Norms and biases are a few numbers each and stay as they are."""
    return [k for k, v in model.state_dict().items() if v.dim() == 2]


def size_bytes(model, bits: int, group) -> int:
    """The bytes of the model with its matrices in `bits` bits and a 16-bit scale per group (group "row": per row; a
    row that does not split into whole groups has a shorter last one), and everything else in 32 bits. 32 bits is the
    model as it is."""
    total = 0
    for k, v in model.state_dict().items():
        if bits == 32 or v.dim() != 2:
            total += 4 * v.numel()
        else:
            rows, cols = v.shape
            per = cols if group == "row" else group
            total += v.numel() * bits // 8 + rows * -(-cols // per) * SCALE_BITS // 8
    return total


def quantized(model, quantize, bits: int, group):
    """A copy of the model whose matrices are what `quantize` makes of them: what the model computes with after
    storing them in `bits` bits. A row that does not split into whole groups, such as a row of 341 weights of mlp.down
    in a model with d = 128, gets zeros at its end until it does; they are cut off again after."""
    import torch
    out = copy.deepcopy(model)
    state = out.state_dict()
    with torch.no_grad():
        for k in matrices(model):
            w = state[k]
            rows, cols = w.shape
            size = cols if group == "row" else group
            padded = torch.cat([w, w.new_zeros(rows, -cols % size)], dim=1)   # always a copy, which quantize may change
            new = quantize(padded, bits, size)
            if not isinstance(new, torch.Tensor) or new.shape != padded.shape or not torch.isfinite(new).all():
                raise ValueError(k)
            state[k] = new[:, :cols].to(w.dtype)
    out.load_state_dict(state)
    return out


def picture_group() -> list[float]:
    """The GROUP weights of the picture, from the course's model."""
    import torch
    state = torch.load((paths.ROOT / P.COURSE).with_suffix(".pt"), map_location="cpu", weights_only=True)
    name, row = PICTURE
    return [float(v) for v in state[name][row, :GROUP].float()]


def group_picture(bits: int, labels: dict) -> str:
    """The GROUP weights of one group of the course's model as hollow rings, and where `bits` bits put them as dots, in
    the colour of what a model learns; with 31 levels or fewer, the levels as thin lines."""
    from everything_ai import chart
    import torch
    ws = picture_group()
    q = course_quantize(torch.tensor([ws]), bits, GROUP)[0].tolist()
    top = max(abs(v) for v in ws)
    p = chart.Panel((-1, GROUP), (-top * 1.15, top * 1.15), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    levels = 2 * qmax(bits) + 1
    if levels <= 31:
        step = top / qmax(bits)
        for k in range(-qmax(bits), qmax(bits) + 1):
            p.line([-1, GROUP], [k * step, k * step], "ink-3", 0.6)
    xs = list(range(GROUP))
    p.points(xs, ws, "ink", r=4.2, hollow=True)
    p.points(xs, q, "param", r=2.6)
    return chart.figure([p], alt=labels["alt"], width=360, height=250)


def group_numbers(bits: int) -> dict:
    """What the picture says at `bits` bits: the levels, the step between two of them, and the largest error."""
    import torch
    ws = picture_group()
    q = course_quantize(torch.tensor([ws]), bits, GROUP)[0].tolist()
    return {"levels": 2 * qmax(bits) + 1, "step": max(abs(v) for v in ws) / qmax(bits),
            "error": max(abs(a - b) for a, b in zip(ws, q))}


def demo(code: str, show=None) -> dict | str:
    """The learner's quantize on their model of llm-pretrain-capstone (or the course's) at each of SETTINGS: its size,
    its loss on the stories, and a story from the model at 4 bits and at 2 bits. "run" when the model or its tools stop
    with an error; "quantize" when the learner's quantize stops with one, or gives back weights of another shape or that
    are not numbers."""
    try:
        model, info, where, ids = model_and_ids()
        tools = run_code("from llm.encode import encode\nfrom llm.decode import decode").names
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    except Exception:
        return "run"
    ran = run_code(code)
    quantize = ran.names.get("quantize") if ran.error is None else None
    if quantize is None:
        return "quantize"
    rows, stories = [], {}
    for bits, group in SETTINGS:
        if show is not None:
            show(bits, group)
        try:
            m = model if bits == 32 else quantized(model, quantize, bits, group)
        except Exception:
            return "quantize"
        rows.append({"bits": bits, "group": group, "bytes": size_bytes(model, bits, group), "loss": val_loss(m, ids)})
        if (bits, group) in ((4, GROUP), (2, GROUP)):
            stories[bits] = P.write(m, merges, tools["encode"], tools["decode"])
    return {"rows": rows, "stories": stories, "source": where, "params": sum(p.numel() for p in model.parameters())}


def demo_picture(rows: list[dict], labels: dict) -> str:
    """The loss on the stories against the bits of each weight, for the groups of GROUP: purple, as it counts the model's
    mistakes; the model as it is at 32 bits, as a dashed line."""
    from everything_ai import chart
    ours = [r for r in rows if r["group"] == GROUP]
    full = next(r["loss"] for r in rows if r["bits"] == 32)
    hi = max(r["loss"] for r in ours)
    p = chart.Panel((1.5, 8.5), (full - 0.6, max(hi, full) + 0.3), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[2, 3, 4, 8])
    p.line([1.5, 8.5], [full, full], "ink-3", 1.2, dashed=True)
    p.text(8.4, full - 0.3, labels["full"], "ink-2", "end")   # below the line: no loss of the chart lies there
    p.line([r["bits"] for r in ours], [r["loss"] for r in ours], "loss", 1.8)
    p.points([r["bits"] for r in ours], [r["loss"] for r in ours], "loss", r=4)
    return chart.figure([p], alt=labels["alt"], width=360, height=250)


# --- measured: the course's model at each of SETTINGS, with the course's quantize, on the first TOKENS ids of the
# stories; on the CPU. tests/test_llm_quantize.py measures them again. ---
MEASURED = {(32, None): 2.7362, (8, GROUP): 2.7365, (4, GROUP): 2.7711, (4, "row"): 2.8098, (3, GROUP): 2.9404,
            (2, GROUP): 6.0343, (4, "matrix"): 3.247}   # "matrix": one scale for a whole matrix


def course_sizes() -> dict:
    """The megabytes of the course's model at each of SETTINGS, as size_bytes counts them."""
    model, _, _ = P.load("course")
    return {(bits, group): size_bytes(model, bits, group) / 1e6 for bits, group in SETTINGS}


# --- exercise 1: quantize ----------------------------------------------------------------------------------------------

_Q_HEAD = ("import torch\n\n\n"
           "def quantize(w, bits, group):\n"
           "    # w: a matrix of weights, each row cut into groups of `group` weights. Each group gets a\n"
           "    # scale, its largest |weight| / qmax; each weight becomes a whole number q from -qmax to\n"
           "    # qmax, with qmax = 2 ** (bits - 1) - 1, and the model computes with q * scale.\n"
           "    # A group of zeros stays zero. The result has the shape of w.\n"
           "    qmax = 2 ** (bits - 1) - 1\n"
           "    rows, cols = w.shape\n"
           "    groups = w.reshape(rows, cols // group, group)\n")
_Q_BODY = ("    scale = groups.abs().amax(dim=2, keepdim=True) / qmax\n"
           "    scale[scale == 0] = 1.0\n"
           "    q = torch.round(groups / scale).clamp(-qmax, qmax)\n"
           "    return (q * scale).reshape(rows, cols)\n")
_Q_START = (_Q_HEAD + "    # 1. the scale of every group, and 1 for a group of zeros\n    ...\n"
            "    # 2. the whole numbers q, and what the model computes with, in the shape of w\n    ...\n")
Q_CODE = _Q_HEAD + _Q_BODY
_Q_WRONG = {
    "unsigned": Q_CODE.replace("    qmax = 2 ** (bits - 1) - 1\n", "    qmax = 2 ** bits - 1\n"),
    "no-round": Q_CODE.replace("torch.round(groups / scale)", "(groups / scale)"),
    "truncate": Q_CODE.replace("torch.round(groups / scale)", "torch.trunc(groups / scale)"),
    "whole": Q_CODE.replace("groups.abs().amax(dim=2, keepdim=True) / qmax", "w.abs().max() / qmax + 0 * groups[:, :, :1]"),
    "row": Q_CODE.replace("groups.abs().amax(dim=2, keepdim=True) / qmax",
                          "w.abs().amax(dim=1).reshape(rows, 1, 1) / qmax + 0 * groups[:, :, :1]"),
    "zero": Q_CODE.replace("    scale[scale == 0] = 1.0\n", ""),
    "shape": Q_CODE.replace("    return (q * scale).reshape(rows, cols)\n", "    return q * scale\n"),
    "keepdim": Q_CODE.replace("amax(dim=2, keepdim=True)", "amax(dim=2)"),
}
_SHOW = "\nquantize(w, {bits}, {group}).tolist()"
_W = "import torch\nw = torch.tensor([[0.8, -0.2, 0.05, 0.3], [0.1, -0.4, 0.6, -0.12]])"
_Q_SCENARIOS = [
    # the lesson's example: 4 bits, groups of 2
    _W + _SHOW.format(bits=4, group=2),
    # one scale per row: a group as long as the row
    _W + _SHOW.format(bits=4, group=4),
    # 2 bits: only -1, 0 and 1 times the scale
    _W + _SHOW.format(bits=2, group=2),
    # 8 bits: close to every weight
    _W + _SHOW.format(bits=8, group=2),
    # a group of zeros stays zero, and the rest is not touched by it
    "import torch\nw = torch.tensor([[0.0, 0.0, 0.5, -0.25]])" + _SHOW.format(bits=3, group=2),
    # the result has the shape of w
    "import torch\nw = torch.randn(4, 6, generator=torch.Generator().manual_seed(0))\nquantize(w, 4, 3).shape",
]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_SAY = ("With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`",
        "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn")

EXERCISES = {
    "quantize": ScenarioBox(
        name="quantize",
        start=_Q_START,
        solution=Q_CODE,
        scenarios=_Q_SCENARIOS,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("unsigned", wrong=_Q_WRONG["unsigned"],
                    en=_SAY[0] + ": your whole numbers run from `-(2 ** bits - 1)` to `2 ** bits - 1`. That is twice as many levels as `bits` bits can hold, and the sign needs a bit of its own. With `bits` bits, q runs from `-qmax` to `qmax`, with `qmax = 2 ** (bits - 1) - 1`: from −7 to 7 for 4 bits.",
                    nl=_SAY[1] + ": je hele getallen lopen van `-(2 ** bits - 1)` tot `2 ** bits - 1`. Dat zijn twee keer zoveel niveaus als `bits` bits kunnen bevatten, en het teken heeft een eigen bit nodig. Met `bits` bits loopt q van `-qmax` tot `qmax`, met `qmax = 2 ** (bits - 1) - 1`: van −7 tot 7 voor 4 bits."),
            Mistake("no-round", wrong=_Q_WRONG["no-round"],
                    en=_SAY[0] + ": your weights come back as they were. Dividing by the scale and multiplying by it again changes nothing; storing in few bits means each weight becomes a whole number first. Round: `torch.round(groups / scale)`.",
                    nl=_SAY[1] + ": je weights komen terug zoals ze waren. Delen door de schaal en er weer mee vermenigvuldigen verandert niets; opslaan in weinig bits betekent dat elke weight eerst een heel getal wordt. Rond af: `torch.round(groups / scale)`."),
            Mistake("truncate", wrong=_Q_WRONG["truncate"],
                    en=_SAY[0] + ": your whole numbers drop what comes after the decimal point, so 2.9 becomes 2 instead of 3. Rounding to the nearest keeps every error below half a step: `torch.round`.",
                    nl=_SAY[1] + ": je hele getallen laten weg wat na de komma komt, dus 2,9 wordt 2 in plaats van 3. Afronden op het dichtstbijzijnde houdt elke fout onder een halve stap: `torch.round`."),
            Mistake("whole", wrong=_Q_WRONG["whole"],
                    en=_SAY[0] + ": your scale is that of the whole matrix, its largest |weight| / qmax, for every group. Then one large weight makes the steps of every group large. Give each group its own scale: the largest |weight| of that group, `groups.abs().amax(dim=2, keepdim=True)`, divided by qmax.",
                    nl=_SAY[1] + ": je schaal is die van de hele matrix, zijn grootste |weight| / qmax, voor elke groep. Dan maakt één grote weight de stappen van elke groep groot. Geef elke groep een eigen schaal: de grootste |weight| van die groep, `groups.abs().amax(dim=2, keepdim=True)`, gedeeld door qmax."),
            Mistake("row", wrong=_Q_WRONG["row"],
                    en=_SAY[0] + ": your scale is that of the whole row, for every group of it. Give each group of `group` weights its own scale: the largest |weight| of that group, `groups.abs().amax(dim=2, keepdim=True)`, divided by qmax.",
                    nl=_SAY[1] + ": je schaal is die van de hele rij, voor elke groep ervan. Geef elke groep van `group` weights een eigen schaal: de grootste |weight| van die groep, `groups.abs().amax(dim=2, keepdim=True)`, gedeeld door qmax."),
            Mistake("zero", wrong=_Q_WRONG["zero"], state="almost",
                    en=_SAY[0] + ": a group of zeros has a scale of 0, and `0 / 0` is `nan`, not a number. A model with one `nan` weight soon gives `nan` everywhere. Give such a group a scale of 1: `scale[scale == 0] = 1.0`; its weights stay 0.",
                    nl=_SAY[1] + ": een groep nullen heeft een schaal van 0, en `0 / 0` is `nan`, geen getal. Een model met één `nan`-weight geeft al snel overal `nan`. Geef zo'n groep een schaal van 1: `scale[scale == 0] = 1.0`; zijn weights blijven 0."),
            Mistake("python-round", error=lambda e: isinstance(e, TypeError) and "__round__" in str(e),
                    en="Python's own `round` works on one number, and `groups / scale` is a tensor of many. Use `torch.round(groups / scale)`: it rounds every number of the tensor at once.",
                    nl="Pythons eigen `round` werkt op één getal, en `groups / scale` is een tensor met veel getallen. Gebruik `torch.round(groups / scale)`: die rondt elk getal van de tensor in één keer af."),
            Mistake("keepdim", wrong=_Q_WRONG["keepdim"],
                    en=_SAY[0] + ": your scales have lost the last axis, so `groups / scale` divides by the wrong scale, or by all of them at once. Keep one number per group as an axis of length 1: `amax(dim=2, keepdim=True)`.",
                    nl=_SAY[1] + ": je schalen zijn de laatste as kwijt, dus `groups / scale` deelt door de verkeerde schaal, of door alle tegelijk. Houd één getal per groep als een as van lengte 1: `amax(dim=2, keepdim=True)`."),
            Mistake("shape", wrong=_Q_WRONG["shape"],
                    en=_SAY[0] + ": your result has the shape of the groups, not of `w`. The model needs its matrix back as it was: `.reshape(rows, cols)`.",
                    nl=_SAY[1] + ": je resultaat heeft de vorm van de groepen, niet die van `w`. Het model heeft zijn matrix nodig zoals hij was: `.reshape(rows, cols)`."),
        ],
        texts={
            **P.S._TEXTS,
            "empty": {"en": "Replace the two `...`: the scale of every group, then the whole numbers q and the weights the model computes with.",
                      "nl": "Vervang de twee `...`: de schaal van elke groep, dan de hele getallen q en de weights waarmee het model rekent."},
            "wrong": {"en": _SAY[0] + ". Does every group get the scale of its largest |weight| / qmax, does every weight become `round(weight / scale)` from `-qmax` to `qmax`, and does the result come back as `q * scale` in the shape of `w`?",
                      "nl": _SAY[1] + ". Krijgt elke groep de schaal van zijn grootste |weight| / qmax, wordt elke weight `round(weight / scale)` van `-qmax` tot `qmax`, en komt het resultaat terug als `q * scale` in de vorm van `w`?"},
            "almost": {"en": "Right for the lesson's example. But " + _SAY[0][0].lower() + _SAY[0][1:] + ". Does it work for one scale per row, for 2 and 8 bits, for a group of zeros, and does the result keep the shape of `w`?",
                       "nl": "Goed voor het voorbeeld van de les. Maar " + _SAY[1][0].lower() + _SAY[1][1:] + ". Werkt hij voor één schaal per rij, voor 2 en 8 bits, voor een groep nullen, en houdt het resultaat de vorm van `w`?"},
            "correct": {"en": "Your `quantize` stores every group of weights in `bits` bits with a scale of its own. Below, it runs on every matrix of your model.",
                        "nl": "Je `quantize` slaat elke groep weights op in `bits` bits met een eigen schaal. Hieronder draait hij op elke matrix van je model."},
        },
    ),
}

# The lesson is done when these are correct. "memory" is a choice question, "faster" the open question at the end.
REQUIRED = ["quantize", "memory", "faster"]

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
CASES = [
    ("quantize", _Q_START, "empty", None),
    ("quantize", Q_CODE, "correct", None),
    ("quantize", _Q_WRONG["unsigned"], "wrong", "unsigned"),
    ("quantize", _Q_WRONG["no-round"], "wrong", "no-round"),
    ("quantize", _Q_WRONG["truncate"], "wrong", "truncate"),
    ("quantize", _Q_WRONG["whole"], "wrong", "whole"),
    ("quantize", _Q_WRONG["row"], "wrong", "row"),
    ("quantize", _Q_WRONG["zero"], "almost", "zero"),
    ("quantize", _Q_WRONG["shape"], "wrong", "shape"),
    ("quantize", _Q_WRONG["keepdim"], "wrong", "keepdim"),
    ("quantize", Q_CODE.replace("torch.round(groups / scale)", "round(groups / scale)"), "error", "python-round"),
    ("quantize", _Q_HEAD + "    scale = ...\n    ...\n", "wrong", "dots-left"),
    # other right ways: a loop over the groups, and torch.where for the zeros
    ("quantize", _Q_HEAD + "    out = groups.clone()\n    for i in range(rows):\n        for j in range(cols // group):\n"
     "            g = groups[i, j]\n            s = g.abs().max() / qmax\n            if s > 0:\n"
     "                out[i, j] = torch.round(g / s).clamp(-qmax, qmax) * s\n    return out.reshape(rows, cols)\n", "correct", None),
    ("quantize", _Q_HEAD + "    top = groups.abs().amax(dim=2, keepdim=True)\n"
     "    scale = torch.where(top > 0, top / qmax, torch.ones_like(top))\n"
     "    return (torch.round(groups / scale) * scale).reshape(rows, cols)\n", "correct", None),
]
