# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-lora: LoRALinear, a frozen linear layer with a low-rank update B @ A that starts at zero
and can be switched off; add_lora, a LoRALinear around every linear layer of the blocks of a ModernGPT, with the rest
frozen; merge_lora, the update merged into one plain linear layer; and a choice. The demo fine-tunes the learner's model
of llm-pretrain-capstone (or the course's) as in llm-sft, but only its adapters, and compares it with full SFT; it keeps
the adapters in work/projects/llm/lora.pt."""
import importlib.util
import json
import time

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _checks("llm-sft", "everything_ai_checks_llm_sft_for_lora")
P, S = T.P, T.S
SAVES = (("llm", "lora", "llm-lora", "lora-linear"), ("llm", "add_lora", "llm-lora", "add-lora"),
         ("llm", "merge_lora", "llm-lora", "merge-lora"))
BUILDS_ON = T.BUILDS_ON + T.SAVES + SAVES   # add_lora and merge_lora import lora; the demo imports all three
projects.builds_on(*BUILDS_ON)

RANKS = (1, 2, 4, 8, 16)   # the ranks the demo offers
RANK = 8                   # the rank the course measured, with alpha = 2 * rank: a scale of 2 at every rank
LR = 5e-3                  # five times the learning rate of full SFT: the adapters start at zero and are few
STEPS = T.STEPS
NAME = "lora"              # the adapters in work/projects/llm/: lora.pt, with lora.json


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def lora_picture(d: int, r: int, labels: dict):
    """A frozen d × d weight W, in grey, and its update B @ A of rank r, in the colour of what a model learns: B a tall
    thin matrix, A a wide thin one, drawn to scale."""
    import html
    from everything_ai import draw
    s = 150 / d   # pixels per number
    big, thin = d * s, max(4.0, r * s)
    font = "font-size:13px;fill:var(--ea-ink-2)"
    sign = "font-size:22px;fill:var(--ea-ink);text-anchor:middle"
    y0 = 34
    xb = big + 40
    xa = xb + thin + 14
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
             f'<rect x="0" y="{y0}" width="{big}" height="{big}" style="fill:var(--ea-data-wash);stroke:var(--ea-line);stroke-width:1"/>',
             f'<text x="{big / 2}" y="{y0 + big / 2 + 5}" style="font-size:15px;fill:var(--ea-ink);text-anchor:middle">W</text>',
             f'<text x="{big + 20}" y="{y0 + big / 2 + 7}" style="{sign}">+</text>',
             f'<rect x="{xb}" y="{y0}" width="{thin}" height="{big}" style="fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"/>',
             f'<text x="{xb + thin / 2}" y="{y0 + big + 18}" style="font-size:15px;fill:var(--ea-param-text);text-anchor:middle">B</text>',
             f'<rect x="{xa}" y="{y0}" width="{big}" height="{thin}" style="fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"/>',
             f'<text x="{xa + big / 2}" y="{y0 + thin + 18}" style="font-size:15px;fill:var(--ea-param-text);text-anchor:middle">A</text>',
             f'<text x="0" y="{y0 + big + 42}" style="{font}">{html.escape(labels["w"])}</text>',
             f'<text x="0" y="{y0 + big + 62}" style="font-size:13px;fill:var(--ea-param-text)">{html.escape(labels["ba"])}</text>']
    width = int(xa + big + 2)
    return draw._svg(width, int(y0 + big + 70), labels["alt"], "".join(parts), natural=True)


def small(x: float, lang: str) -> str:
    """A small number with one significant digit, never in e-notation: 0.00001."""
    from decimal import Decimal
    text = format(Decimal(f"{x:.1g}"), "f")
    return text.replace(".", ",") if lang == "nl" else text


def trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tools() -> dict | str:
    """The learner's modules the demo needs, or "run" when one stops with an error."""
    ran = run_code("from llm.fast_encode import encode_fast\nfrom llm.decode import decode\n"
                   "from llm.pretrain_step import pretrain_step\nfrom llm.sft_example import sft_example\n"
                   "from llm.sft_batch import sft_batch\nfrom llm.words_used import words_used\n"
                   "from llm.add_lora import add_lora\nfrom llm.merge_lora import merge_lora")
    return "run" if ran.error is not None else ran.names


def run(rank: int, show) -> dict | str:
    """The demo: the learner's model of the capstone, frozen, with their adapters of `rank` (alpha 2 * rank) around
    every linear layer of its blocks, fine-tuned as in llm-sft but at LR; then its answers to the held-out
    instructions, and the merged model's logits against the model with adapters. The adapters are kept in
    work/projects/llm/lora.pt. A string when it stops: "run", "lora", "memory" or "diverged"."""
    import torch
    t = tools()
    if t == "run":
        return "run"
    try:
        model, info, source = P.load()
        full = sum(p.numel() for p in model.parameters())
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
        end = 256 + len(merges)
        made = T.pairs(T.stories()[:-T.HELD], merges, t["encode_fast"], t["decode"])
        examples = [t["sft_example"](list(prompt), list(answer), end) for _, prompt, answer in made]
    except Exception:
        return "run"
    show({"phase": "pairs", "pairs": len(examples)})
    try:
        probe = torch.tensor([examples[0][0][:16]])
        with torch.no_grad():
            before = model(probe)
        torch.manual_seed(0)   # the same random A on every run
        layers = t["add_lora"](model, rank, 2 * rank)
        with torch.no_grad():
            same = torch.allclose(model(probe), before, atol=1e-5)
    except Exception:
        return "lora"
    if not same or trainable(model) == 0:
        return "lora"
    n_train = trainable(model)
    tuned = T.tune(model, examples, t["sft_batch"], t["pretrain_step"], end, show, lr=LR)
    if isinstance(tuned, str):
        return "run" if tuned == "batch" else tuned
    after = T.rate(model, merges, t["words_used"], t["encode_fast"], t["decode"], show, "after")
    if isinstance(after, str):
        return "run"
    try:
        model.cpu()
        with torch.no_grad():
            with_adapters = model(probe)
            for layer in layers:
                layer.on = False
            off = model(probe)
            for layer in layers:
                layer.on = True
            merged = _merged(model, t["merge_lora"])
            gap = float((merged(probe) - with_adapters).abs().max())
            moved = float((off - with_adapters).abs().max())
    except Exception:
        return "lora"
    result = {"model": model, "source": source, "rank": rank, "alpha": 2 * rank, "params": full, "trainable": n_train,
              "layers": len(layers), "after": after, "gap": gap, "moved": moved, **tuned}
    save(result)
    return result


def _merged(model, merge_lora):
    """A copy of `model` with every LoRALinear merged into a plain linear layer by the learner's merge_lora."""
    import copy
    out = copy.deepcopy(model)
    for block in out.blocks:
        for part, names in ((block.attention, ("query", "key", "value", "out")), (block.mlp, ("gate", "up", "down"))):
            for name in names:
                layer = getattr(part, name)
                if hasattr(layer, "base"):
                    setattr(part, name, merge_lora(layer))
    return out


def save(result: dict) -> None:
    """The adapters, A and B of every LoRALinear, in work/projects/llm/lora.pt, and the run in lora.json: also, for
    every test instruction in order, 1 when the answer used every word and 0 when not."""
    import torch
    folder = projects.folder() / "llm"
    folder.mkdir(parents=True, exist_ok=True)
    adapters = {k: v.detach().cpu() for k, v in result["model"].state_dict().items() if k.endswith((".A", ".B"))}
    torch.save(adapters, folder / f"{NAME}.pt")
    keep = {"base": result["source"], "rank": result["rank"], "alpha": result["alpha"], "params": result["params"],
            "trainable": result["trainable"], "layers": result["layers"], "steps": result["steps"], "lr": LR,
            "seconds": round(result["seconds"], 1), "device": result["device"], "losses": result["losses"],
            "after": {k: result["after"][k] for k in ("used", "stopped", "total")}, "gap": result["gap"],
            "hits": "".join("1" if h else "0" for h in result["after"]["hits"]),
            "bytes": (folder / f"{NAME}.pt").stat().st_size}
    (folder / f"{NAME}.json").write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")


def saved() -> dict | None:
    folder = projects.folder() / "llm"
    try:
        info = json.loads((folder / f"{NAME}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if (folder / f"{NAME}.pt").is_file() else None


def full_sft(base: str | None = None) -> dict:
    """What full SFT gave on the model a LoRA run fine-tuned: the learner's run of llm-sft when it fine-tuned that
    model (`base` "yours" or "course"), else the course's measured run, on its model of the capstone's demo run."""
    mine = T.saved() if base is not None else None
    if mine is not None and mine.get("base") == base:
        return {"source": "yours", "params": mine["params"], "used": mine["after"]["used"], "total": mine["after"]["total"],
                "seconds": mine["seconds"], "before": mine["before"]["used"]}
    m = T.MEASURED["yours"]
    return {"source": None, "params": m["params"], "used": m["after"]["used"], "total": m["after"]["total"],
            "seconds": m["seconds"], "before": m["before"]["used"]}


# --- measured: the course's run of the demo with its own solutions, on the CPU, one thread, on its model of the demo
# run of llm-pretrain-capstone (TinyStories, so not shared: only its numbers); "seconds" the steps, "other_seconds" the
# examples, the answers and the merge; "gap" the largest difference between the logits merged and with adapters ---
MEASURED = {
    "yours": {"rank": RANK, "d": 96, "params": 575337, "trainable": 43776, "layers": 21, "seconds": 198.5, "other_seconds": 71.1,
              "after": {"used": 35, "stopped": 100, "total": T.TESTS}, "bytes": 185365, "gap": 1.0e-05,
              # for every test instruction in order, 1 when the answer used every word
              "hits": "1000001110001000010010000010001000100000001110111100000101101010101010000010000110000011011010101000"},
}
# --- end of measured ---

REQUIRED = ["lora-linear", "add-lora", "merge-lora", "why-zero"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_SCENARIO, _SCENARIO_NL = T._SCENARIO, T._SCENARIO_NL

# --- exercise 1: LoRALinear --------------------------------------------------------------------------------------------

_L_HEAD = "import torch\nfrom torch import nn\n\n\nclass LoRALinear(nn.Module):\n    def __init__(self, base, r, alpha):\n" \
          "        super().__init__()\n        self.base = base\n"
_L_FREEZE = "        for p in self.base.parameters():\n            p.requires_grad = False\n"
_L_A = "        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)\n"
_L_B = "        self.B = nn.Parameter(torch.zeros(base.out_features, r))\n"
_L_TAIL = "        self.scale = alpha / r\n        self.on = True\n\n    def forward(self, x):\n        out = self.base(x)\n"
_L_UPDATE = "        if self.on:\n            out = out + x @ self.A.T @ self.B.T * self.scale\n"


def _l(freeze=_L_FREEZE, a=_L_A, b=_L_B, update=_L_UPDATE) -> str:
    return _L_HEAD + freeze + a + b + _L_TAIL + update + "        return out\n"


_L_START = (_L_HEAD + "        # freeze the parameters of the layer itself\n        ...\n"
            "        # A: r rows of in_features random numbers; B: out_features rows of r zeros\n"
            "        self.A = ...\n        self.B = ...\n"
            + _L_TAIL + "        # the low-rank update, when it is on\n        ...\n        return out\n")
_L_SETUP = "torch.manual_seed(0)\nbase = nn.Linear(6, 4)\nx = torch.randn(3, 6)\nlayer = LoRALinear(base, 2, 4)\n"
_L_SET = "with torch.no_grad():\n    layer.A.copy_(torch.arange(12.0).reshape(2, 6) / 10)\n    layer.B.copy_(torch.ones(4, 2))\n"
_L_SCENARIOS = [
    # the lesson's example: a new layer gives what the base gives
    _L_SETUP + "torch.allclose(layer(x), base(x))",
    # only A and B learn, in their shapes
    _L_SETUP + "sorted((name, tuple(p.shape)) for name, p in layer.named_parameters() if p.requires_grad)",
    # A starts random, of rank r; B at zero
    "import numpy as np\n" + _L_SETUP + "int(np.linalg.matrix_rank(layer.A.detach().numpy())), bool((layer.B == 0).all())",
    # with A and B set: the update, scaled by alpha / r
    _L_SETUP + _L_SET + "[[round(v, 4) for v in row] for row in (layer(x) - base(x)).tolist()]",
    # switched off, the layer is the base again
    _L_SETUP + _L_SET + "layer.on = False\ntorch.allclose(layer(x), base(x))",
    # a gradient reaches B at once, none reaches the base
    _L_SETUP + "layer(x).sum().backward()\nlayer.B.grad is not None and bool(layer.B.grad.abs().sum() > 0), base.weight.grad is None",
    # the frozen base still passes a gradient back to its input, for the layers before it
    _L_SETUP + "x.requires_grad_(True)\nlayer(x).sum().backward()\nx.grad is not None and torch.allclose(x.grad, base.weight.sum(0).expand(3, 6))",
    # a batch of windows, as in a model
    _L_SETUP + "tuple(layer(torch.randn(2, 5, 6)).shape)",
]
_L_B_RANDOM = _l(b="        self.B = nn.Parameter(torch.randn(base.out_features, r))\n")
_L_A_ZERO = _l(a="        self.A = nn.Parameter(torch.zeros(r, base.in_features))\n")
_L_A_SAME = _l(a="        self.A = nn.Parameter(torch.ones(r, base.in_features) / base.in_features ** 0.5)\n")
_L_NOT_FROZEN = _l(freeze="")
_L_NO_SCALE = _l(update=_L_UPDATE.replace(" * self.scale", ""))
_L_ALWAYS = _l(update="        out = out + x @ self.A.T @ self.B.T * self.scale\n")
_L_NO_GRAD = (_L_HEAD + _L_FREEZE + _L_A + _L_B + _L_TAIL.replace("        out = self.base(x)\n", "        with torch.no_grad():\n            out = self.base(x)\n")
              + _L_UPDATE + "        return out\n")

# --- exercise 2: add_lora ----------------------------------------------------------------------------------------------

_D_HEAD = "from llm.lora import LoRALinear\n\n\ndef add_lora(model, r, alpha):\n"
_D_FREEZE = "    for p in model.parameters():\n        p.requires_grad = False\n"
_D_LOOP = ("    layers = []\n    for block in model.blocks:\n        a, m = block.attention, block.mlp\n")
_D_WRAP = ("        a.query = LoRALinear(a.query, r, alpha)\n        a.key = LoRALinear(a.key, r, alpha)\n"
           "        a.value = LoRALinear(a.value, r, alpha)\n        a.out = LoRALinear(a.out, r, alpha)\n"
           "        m.gate = LoRALinear(m.gate, r, alpha)\n        m.up = LoRALinear(m.up, r, alpha)\n"
           "        m.down = LoRALinear(m.down, r, alpha)\n")
_D_KEEP = "        layers += [a.query, a.key, a.value, a.out, m.gate, m.up, m.down]\n    return layers\n"
_D = _D_HEAD + _D_FREEZE + _D_LOOP + _D_WRAP + _D_KEEP
_D_START = (_D_HEAD + "    # freeze every parameter of the model\n    ...\n" + _D_LOOP
            + "        # a LoRALinear around a.query, a.key, a.value and a.out, and around m.gate, m.up and m.down\n        ...\n"
            + _D_KEEP)
_D_SETUP = ("import torch\nfrom llm.modern_gpt import ModernGPT\n\ntorch.manual_seed(0)\nmodel = ModernGPT(40, 16, 2, 1, 2)\n"
            "ids = torch.tensor([[3, 1, 4, 1, 5]])\nbefore = model(ids)\nlayers = add_lora(model, 2, 4)\n")
_D_SCENARIOS = [
    # the lesson's example: the layers, and the numbers that learn
    _D_SETUP + "len(layers), sum(p.numel() for p in model.parameters() if p.requires_grad)",
    # only the A and B of the adapters learn
    _D_SETUP + "sorted({name.rsplit('.', 1)[-1] for name, p in model.named_parameters() if p.requires_grad})",
    # the model gives what it gave before
    _D_SETUP + "torch.allclose(model(ids), before)",
    # every block has its adapters
    _D_SETUP + "[type(b.attention.query).__name__ for b in model.blocks], [type(b.mlp.down).__name__ for b in model.blocks]",
]
_D_NOT_FROZEN = _D_HEAD + _D_LOOP + _D_WRAP + _D_KEEP
_D_FREEZE_AFTER = _D_HEAD + _D_LOOP + _D_WRAP + "        layers += [a.query, a.key, a.value, a.out, m.gate, m.up, m.down]\n" + _D_FREEZE + "    return layers\n"
_D_ATTENTION = _D_HEAD + _D_FREEZE + _D_LOOP + _D_WRAP.split("        m.gate")[0] + "        layers += [a.query, a.key, a.value, a.out]\n    return layers\n"

# --- exercise 3: merge_lora --------------------------------------------------------------------------------------------

_M_HEAD = "import torch\nfrom torch import nn\n\n\ndef merge_lora(layer):\n"
_M = (_M_HEAD + "    base = layer.base\n"
      "    merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)\n"
      "    with torch.no_grad():\n"
      "        merged.weight.copy_(base.weight + layer.scale * layer.B @ layer.A)\n"
      "        if base.bias is not None:\n"
      "            merged.bias.copy_(base.bias)\n"
      "    return merged\n")
_M_START = (_M_HEAD + "    base = layer.base\n"
            "    merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)\n"
            "    with torch.no_grad():\n"
            "        # the weight of the base plus the update, scaled\n        ...\n"
            "        # and the bias of the base, when it has one\n        ...\n"
            "    return merged\n")
_M_SETUP = ("from llm.lora import LoRALinear\n\ntorch.manual_seed(0)\nlayer = LoRALinear(nn.Linear(6, 4), 2, 4)\n"
            "with torch.no_grad():\n    layer.B.copy_(torch.randn(4, 2))\nx = torch.randn(3, 6)\n")
_M_SCENARIOS = [
    # the lesson's example: the merged layer gives what the layer with its update gives
    _M_SETUP + "merged = merge_lora(layer)\nround(float((merged(x) - layer(x)).abs().max()), 4)",
    # a plain linear layer, as big as the base
    _M_SETUP + "merged = merge_lora(layer)\ntype(merged).__name__, tuple(merged.weight.shape)",
    # the base is not changed by merging
    _M_SETUP + "w = layer.base.weight.clone()\nmerged = merge_lora(layer)\ntorch.equal(layer.base.weight, w)",
    # a base without a bias
    "from llm.lora import LoRALinear\n\ntorch.manual_seed(1)\nlayer = LoRALinear(nn.Linear(5, 3, bias=False), 2, 2)\n"
    "with torch.no_grad():\n    layer.B.copy_(torch.randn(3, 2))\nx = torch.randn(2, 5)\n"
    "merged = merge_lora(layer)\nmerged.bias is None, torch.allclose(merged(x), layer(x), atol=1e-6)",
]
_M_NO_SCALE = _M.replace("base.weight + layer.scale * layer.B @ layer.A", "base.weight + layer.B @ layer.A")
_M_NO_BIAS = _M.replace("        if base.bias is not None:\n            merged.bias.copy_(base.bias)\n", "")
_M_IN_PLACE = (_M_HEAD + "    base = layer.base\n"
               "    with torch.no_grad():\n"
               "        base.weight += layer.scale * layer.B @ layer.A\n"
               "    return base\n")

EXERCISES = {
    "lora-linear": ClassBox(
        name="lora-linear",
        start=_L_START,
        solution=_l(),
        scenarios=_L_SCENARIOS,
        save_as=("llm", "lora"),
        first=[_DOTS],
        mistakes=[
            Mistake("shapes", error=lambda e: "cannot be multiplied" in str(e) or "size of tensor" in str(e),
                    en="The shapes of `x`, `A` and `B` do not fit together. `A` has `r` rows of `base.in_features` numbers, `B` has `base.out_features` rows of `r`, and the update is `x @ self.A.T @ self.B.T`: from `in_features` numbers to `r`, and from `r` to `out_features`.",
                    nl="De vormen van `x`, `A` en `B` passen niet op elkaar. `A` heeft `r` rijen van `base.in_features` getallen, `B` heeft `base.out_features` rijen van `r`, en de update is `x @ self.A.T @ self.B.T`: van `in_features` getallen naar `r`, en van `r` naar `out_features`."),
            Mistake("b-random", wrong=_L_B_RANDOM,
                    en=_SCENARIO + ": `B` does not start at zero, so a new layer already changes what the model gives. Start `B` at zero, `torch.zeros(base.out_features, r)`: then `B @ A` is zero, and the model starts exactly as it was.",
                    nl=_SCENARIO_NL + ": `B` begint niet op nul, dus verandert een nieuwe layer al wat het model geeft. Laat `B` op nul beginnen, `torch.zeros(base.out_features, r)`: dan is `B @ A` nul, en begint het model precies zoals het was."),
            Mistake("a-zero", wrong=_L_A_ZERO,
                    en=_SCENARIO + ": `A` starts at zero too. Then neither ever gets a gradient other than zero: the gradient of `B` goes through `A`, and that of `A` through `B`. Start `A` random: `torch.randn(r, base.in_features) / base.in_features ** 0.5`.",
                    nl=_SCENARIO_NL + ": `A` begint ook op nul. Dan krijgt geen van beide ooit een andere gradient dan nul: de gradient van `B` loopt via `A`, en die van `A` via `B`. Laat `A` willekeurig beginnen: `torch.randn(r, base.in_features) / base.in_features ** 0.5`."),
            Mistake("a-rank-one", wrong=_L_A_SAME,
                    en=_SCENARIO + ": `A` starts with rank 1: all its rows are multiples of one row. Every row of `B @ A` is a mix of the rows of `A`, so the update starts with one direction instead of up to `r`. Start `A` random: `torch.randn(r, base.in_features) / base.in_features ** 0.5`.",
                    nl=_SCENARIO_NL + ": `A` begint met rang 1: al zijn rijen zijn veelvouden van één rij. Elke rij van `B @ A` is een mengsel van de rijen van `A`, dus begint de update met één richting in plaats van hooguit `r`. Laat `A` willekeurig beginnen: `torch.randn(r, base.in_features) / base.in_features ** 0.5`."),
            Mistake("not-frozen", wrong=_L_NOT_FROZEN,
                    en=_SCENARIO + ": the weight and the bias of the base learn too, so the layer is no cheaper to train than the base itself. Freeze them first: `p.requires_grad = False` for every `p` of `self.base.parameters()`.",
                    nl=_SCENARIO_NL + ": de weight en de bias van de basis leren ook, dus is de layer niet goedkoper om te trainen dan de basis zelf. Bevries ze eerst: `p.requires_grad = False` voor elke `p` van `self.base.parameters()`."),
            Mistake("no-scale", wrong=_L_NO_SCALE,
                    en=_SCENARIO + ": the update is not scaled. Multiply it by `self.scale`, which is `alpha / r`.",
                    nl=_SCENARIO_NL + ": de update is niet geschaald. Vermenigvuldig hem met `self.scale`, en dat is `alpha / r`."),
            Mistake("always-on", wrong=_L_ALWAYS,
                    en=_SCENARIO + ": the update stays on when `self.on` is False. Add it only `if self.on:`: a later lesson switches the adapters off to get the model as it was.",
                    nl=_SCENARIO_NL + ": de update blijft aan als `self.on` False is. Tel hem alleen op `if self.on:`: een latere les zet de adapters uit om het model te krijgen zoals het was."),
            Mistake("base-no-grad", wrong=_L_NO_GRAD,
                    en=_SCENARIO + ": the output of the base carries no gradient back to `x`, so in a model the adapters of the layers before this one learn nothing through it. Frozen means that the weight of the base gets no gradient, not its input: compute `self.base(x)` as it is, without `torch.no_grad()` or `detach()`.",
                    nl=_SCENARIO_NL + ": de output van de basis geeft geen gradient terug naar `x`, dus in een model leren de adapters van de layers ervoor er niets door. Bevroren betekent dat de weight van de basis geen gradient krijgt, niet zijn input: reken `self.base(x)` gewoon uit, zonder `torch.no_grad()` of `detach()`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the four `...`: freeze the base, make `A` and `B`, and add the update in `forward`.",
                      "nl": "Vervang de vier `...`: bevries de basis, maak `A` en `B`, en tel de update op in `forward`."},
            "wrong": {"en": _SCENARIO + ". A frozen base, `A` random of shape (r, in_features), `B` zero of shape (out_features, r), and in `forward`, when `self.on`, `x @ self.A.T @ self.B.T * self.scale` added to the output of the base?",
                      "nl": _SCENARIO_NL + ". Een bevroren basis, `A` willekeurig met vorm (r, in_features), `B` nul met vorm (out_features, r), en in `forward`, als `self.on`, `x @ self.A.T @ self.B.T * self.scale` opgeteld bij de output van de basis?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Only A and B learning, the scale, the switch, and a batch of windows?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Alleen A en B die leren, de schaal, de schakelaar, en een batch vensters?"},
            "correct": {"en": "Your layer starts as the base, learns only `A` and `B`, and can be switched off. It is in your project now, as `{saved}`.",
                        "nl": "Je layer begint als de basis, leert alleen `A` en `B`, en kan uit. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `LoRALinear` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `LoRALinear` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "add-lora": ScenarioBox(
        name="add-lora",
        start=_D_START,
        solution=_D,
        scenarios=_D_SCENARIOS,
        save_as=("llm", "add_lora"),
        first=[_DOTS],
        mistakes=[
            Mistake("not-frozen", wrong=_D_NOT_FROZEN,
                    en=_SCENARIO + ": the embedding, the norms and the head still learn, so almost every number of the model trains. Freeze every parameter of the model first, before you add the adapters.",
                    nl=_SCENARIO_NL + ": de embedding, de norms en de head leren nog, dus traint bijna elk getal van het model. Bevries eerst elke parameter van het model, voordat je de adapters toevoegt."),
            Mistake("freeze-after", wrong=_D_FREEZE_AFTER,
                    en=_SCENARIO + ": you freeze after adding the adapters, so their `A` and `B` are frozen too, and nothing learns. Freeze the model first, then add the adapters: their `A` and `B` are new, and learn.",
                    nl=_SCENARIO_NL + ": je bevriest na het toevoegen van de adapters, dus zijn hun `A` en `B` ook bevroren, en leert er niets. Bevries eerst het model, en voeg dan de adapters toe: hun `A` en `B` zijn nieuw, en leren."),
            Mistake("attention-only", wrong=_D_ATTENTION,
                    en=_SCENARIO + ": only the four layers of attention get an adapter. Put one around `m.gate`, `m.up` and `m.down` too: seven per block.",
                    nl=_SCENARIO_NL + ": alleen de vier layers van attention krijgen een adapter. Zet er ook een rond `m.gate`, `m.up` en `m.down`: zeven per block."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: freeze the model, and put a LoRALinear around the seven linear layers of the block.",
                      "nl": "Vervang de twee `...`: bevries het model, en zet een LoRALinear rond de zeven linear layers van het block."},
            "wrong": {"en": _SCENARIO + ". First every parameter frozen, then in every block a LoRALinear around `a.query`, `a.key`, `a.value`, `a.out`, `m.gate`, `m.up` and `m.down`?",
                      "nl": _SCENARIO_NL + ". Eerst elke parameter bevroren, dan in elk block een LoRALinear rond `a.query`, `a.key`, `a.value`, `a.out`, `m.gate`, `m.up` en `m.down`?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Every block, and the model unchanged at the start?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Elk block, en het model ongewijzigd bij de start?"},
            "correct": {"en": "Your model is frozen but for its adapters, and still gives what it gave. It is in your project now, as `{saved}`.",
                        "nl": "Je model is bevroren op zijn adapters na, en geeft nog wat het gaf. Het staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `add_lora` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `add_lora` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "merge-lora": ScenarioBox(
        name="merge-lora",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        save_as=("llm", "merge_lora"),
        first=[_DOTS],
        mistakes=[
            Mistake("order", error=lambda e: "cannot be multiplied" in str(e) or "must match the size" in str(e) or "size of tensor" in str(e),
                    en="The shapes do not fit. The update of the weight is `layer.B @ layer.A`: `B` has `out_features` rows of `r`, `A` has `r` rows of `in_features`, so their product has the shape of the weight, (out_features, in_features).",
                    nl="De vormen passen niet. De update van de weight is `layer.B @ layer.A`: `B` heeft `out_features` rijen van `r`, `A` heeft `r` rijen van `in_features`, dus hun product heeft de vorm van de weight, (out_features, in_features)."),
            Mistake("no-scale", wrong=_M_NO_SCALE,
                    en=_SCENARIO + ": the merged weight leaves out the scale. The layer adds `B @ A` times `layer.scale`, so the merged weight is `base.weight + layer.scale * layer.B @ layer.A`.",
                    nl=_SCENARIO_NL + ": de samengevoegde weight laat de schaal weg. De layer telt `B @ A` keer `layer.scale` op, dus de samengevoegde weight is `base.weight + layer.scale * layer.B @ layer.A`."),
            Mistake("no-bias", wrong=_M_NO_BIAS,
                    en=_SCENARIO + ": the merged layer keeps its own random bias. Copy the bias of the base into it, when the base has one.",
                    nl=_SCENARIO_NL + ": de samengevoegde layer houdt zijn eigen willekeurige bias. Kopieer de bias van de basis erin, als de basis er een heeft."),
            Mistake("in-place", wrong=_M_IN_PLACE,
                    en=_SCENARIO + ": you add the update to the weight of the base itself, so the layer with its adapter changes too, and adds the update twice. Make a new `nn.Linear`, and copy the sum into its weight.",
                    nl=_SCENARIO_NL + ": je telt de update op bij de weight van de basis zelf, dus verandert de layer met zijn adapter ook, en telt die de update twee keer op. Maak een nieuwe `nn.Linear`, en kopieer de som in zijn weight."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: copy the merged weight into `merged.weight`, and the bias of the base into `merged.bias`.",
                      "nl": "Vervang de twee `...`: kopieer de samengevoegde weight in `merged.weight`, en de bias van de basis in `merged.bias`."},
            "wrong": {"en": _SCENARIO + ". A new layer with the weight of the base plus `layer.scale` times `B @ A`, and the bias of the base?",
                      "nl": _SCENARIO_NL + ". Een nieuwe layer met de weight van de basis plus `layer.scale` keer `B @ A`, en de bias van de basis?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". A base without a bias, and a base that stays as it was?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Een basis zonder bias, en een basis die blijft zoals hij was?"},
            "correct": {"en": "Your merged layer gives what the layer with its adapter gives, with one weight matrix: no extra work when the model writes. It is in your project now, as `{saved}`.",
                        "nl": "Je samengevoegde layer geeft wat de layer met zijn adapter geeft, met één weight matrix: geen extra werk als het model schrijft. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `merge_lora` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `merge_lora` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("lora-linear", _L_START, "empty", None),
    ("lora-linear", _l(), "correct", None),
    ("lora-linear", _l(a="        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)\n"), "correct", None),
    ("lora-linear", _l(update="        if self.on:\n            out = out + self.scale * ((x @ self.A.T) @ self.B.T)\n"), "correct", None),
    ("lora-linear", _l(update="        if self.on:\n            out = out + (self.B @ self.A @ x.unsqueeze(-1)).squeeze(-1) * self.scale\n"), "correct", None),
    ("lora-linear", _l(a="        self.A = nn.Parameter(torch.empty(r, base.in_features))\n        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)\n"), "correct", None),
    ("lora-linear", _L_HEAD + _L_FREEZE + _L_B + _L_A + _L_TAIL + _L_UPDATE + "        return out\n", "correct", None),
    ("lora-linear", _L_B_RANDOM, "wrong", "b-random"),
    ("lora-linear", _L_A_ZERO, "wrong", "a-zero"),
    ("lora-linear", _L_A_SAME, "wrong", "a-rank-one"),
    ("lora-linear", _l(b="        self.B = nn.Parameter(torch.ones(base.out_features, r))\n"), "wrong", "b-random"),
    ("lora-linear", _L_NOT_FROZEN, "wrong", "not-frozen"),
    ("lora-linear", _L_NO_SCALE, "wrong", "no-scale"),
    ("lora-linear", _L_ALWAYS, "wrong", "always-on"),
    ("lora-linear", _L_NO_GRAD, "wrong", "base-no-grad"),
    ("lora-linear", _l(update="        out = out.detach()\n" + _L_UPDATE), "wrong", "base-no-grad"),
    ("lora-linear", _l(update=_L_UPDATE.replace("x @ self.A.T @ self.B.T", "x @ self.A @ self.B")), "error", "shapes"),
    ("lora-linear", _l(b="        self.B = nn.Parameter(torch.zeros(r, base.out_features))\n"), "error", "shapes"),
    ("add-lora", _D_START, "empty", None),
    ("add-lora", _D, "correct", None),
    ("add-lora", _D_HEAD + _D_FREEZE + "    layers = []\n    for block in model.blocks:\n"
                 "        for part, names in ((block.attention, (\"query\", \"key\", \"value\", \"out\")), (block.mlp, (\"gate\", \"up\", \"down\"))):\n"
                 "            for name in names:\n                layer = LoRALinear(getattr(part, name), r, alpha)\n"
                 "                setattr(part, name, layer)\n                layers.append(layer)\n    return layers\n", "correct", None),
    ("add-lora", _D_NOT_FROZEN, "wrong", "not-frozen"),
    ("add-lora", _D_FREEZE_AFTER, "wrong", "freeze-after"),
    ("add-lora", _D_ATTENTION, "wrong", "attention-only"),
    ("merge-lora", _M_START, "empty", None),
    ("merge-lora", _M, "correct", None),
    ("merge-lora", _M.replace("base.weight + layer.scale * layer.B @ layer.A", "base.weight + (layer.A.T @ layer.B.T).T * layer.scale"), "correct", None),
    ("merge-lora", _M_NO_SCALE, "wrong", "no-scale"),
    ("merge-lora", _M_NO_BIAS, "wrong", "no-bias"),
    ("merge-lora", _M_IN_PLACE, "wrong", "in-place"),
    ("merge-lora", _M.replace("layer.B @ layer.A", "layer.A @ layer.B"), "error", "order"),
]
