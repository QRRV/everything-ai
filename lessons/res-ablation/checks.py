# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-ablation: no_positions, which takes the position embedding out of a GPT. And the demo:
the run of gpt-train-chars once more, with the positions taken out, next to the run with them."""
import html
import importlib.util
import re

from everything_ai import paths, projects
from everything_ai.check import Mistake, ScenarioBox


def _train_chars():
    """The checks of gpt-train-chars, whose run this lesson repeats with one part taken out. Loading them registers
    that lesson's project modules; this lesson's own list is put back afterwards."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_gpt_train_chars_for_ablation",
                                                  paths.LESSONS / "gpt-train-chars" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _train_chars()

# The run uses the learner's GPT and everything gpt-train-chars builds on, and this lesson's no_positions; where one
# is missing or broken, the course's version stands in (template 1.3).
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"),
                   ("llm", "gpt_data", "gpt-assemble", "gpt-examples"),
                   ("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("llm", "gpt_loaders", "gpt-train-chars", "gpt-loaders"), ("llm", "gpt_run", "gpt-train-chars", "gpt-run"),
                   ("llm", "ablate", "res-ablation", "no-positions"))

NAME = "gpt_nopos"   # the files of the run without positions: work/projects/llm/gpt_nopos.pt and .json

# The course's runs of the GPT without positions, on the plans of gpt-train-chars and made in the same way as its
# REFERENCE (the CPU plan with seeds 0 to 2; the GPU plan with seed 0, run on the CPU): the history of each.
REFERENCE = {
    "cpu": [
        {"seed": 0, "history": [
            (1, 2.6376, 2.4135), (2, 2.2694, 2.2627), (3, 2.0238, 2.059), (4, 1.8029, 1.9074), (5, 1.6827, 1.8212),
            (6, 1.6012, 1.7513), (7, 1.5488, 1.71), (8, 1.5076, 1.6851), (9, 1.4829, 1.672), (10, 1.4685, 1.6683),
        ]},
        {"seed": 1, "history": [
            (1, 2.6369, 2.4413), (2, 2.2863, 2.2614), (3, 2.0464, 2.023), (4, 1.822, 1.9061), (5, 1.6893, 1.8089),
            (6, 1.6098, 1.7542), (7, 1.5511, 1.7076), (8, 1.5108, 1.6841), (9, 1.4831, 1.6702), (10, 1.4706, 1.6677),
        ]},
        {"seed": 2, "history": [
            (1, 2.649, 2.4241), (2, 2.2482, 2.2324), (3, 1.9603, 2.0119), (4, 1.7688, 1.8898), (5, 1.6607, 1.8082),
            (6, 1.5856, 1.7439), (7, 1.5269, 1.7003), (8, 1.4917, 1.68), (9, 1.4677, 1.666), (10, 1.4518, 1.6631),
        ]},
    ],
    "gpu": [
        {"seed": 0, "history": [
            (1, 2.7966, 2.496), (2, 2.4497, 2.4571), (3, 2.3879, 2.4276), (4, 2.3135, 2.378), (5, 2.2163, 2.2612),
            (6, 2.0057, 2.0372), (7, 1.7861, 1.8948), (8, 1.6502, 1.7709), (9, 1.5491, 1.7221), (10, 1.4808, 1.671),
            (11, 1.4309, 1.6291), (12, 1.3858, 1.5947), (13, 1.3419, 1.5787), (14, 1.3146, 1.5818), (15, 1.2914, 1.5537),
            (16, 1.2612, 1.5466), (17, 1.241, 1.5398), (18, 1.2204, 1.539), (19, 1.21, 1.5389), (20, 1.2091, 1.5384),
        ]},
    ],
}


def course_differences(plan: str) -> list:
    """The course's runs of a plan, seed by seed: the lowest validation loss without positions minus that with them."""
    with_pos = {r["seed"]: min(h[2] for h in r["history"]) for r in T.REFERENCE.get(plan, [])}
    return [min(h[2] for h in r["history"]) - with_pos[r["seed"]] for r in REFERENCE.get(plan, []) if r["seed"] in with_pos]


def signed(x: float, digits: int, lang: str) -> str:
    """A difference with its sign: +0.007, or −0.011 with a real minus sign."""
    return ("+" if x >= 0 else "\u2212") + number(abs(x), digits, lang)


def _dots(got, text) -> bool:
    return re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None


_DOTS = Mistake("dots-left", matches=_dots,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_HEAD = "import torch\n\n\ndef no_positions(model):\n"
_ZERO = "    with torch.no_grad():\n        model.pos.weight[:] = 0\n"
_FREEZE = "    model.pos.weight.requires_grad = False\n"
_START = (_HEAD + "    # every number of the table of places to 0, without a graph\n    ...\n"
          "    # and the table no longer trains\n    ...\n")


def _a(zero: str = _ZERO, freeze: str = _FREEZE) -> str:
    return _HEAD + zero + freeze


_MODEL = "import torch\nfrom llm.gpt import GPT\n\ntorch.manual_seed(0)\nmodel = GPT(5, 8, 2, 1, 4)\n"
_TRAINABLE = "sum(p.numel() for p in model.parameters() if p.requires_grad)"
_SCENARIOS = [
    # the lesson's example: the table of places afterwards, whether it still trains, how many numbers still train, and
    # the table of tokens, which stays as it was
    _MODEL + "no_positions(model)\n[round(float(model.pos.weight.abs().sum()), 4), model.pos.weight.requires_grad, " + _TRAINABLE
    + ", round(float(model.tok.weight.abs().sum()), 4)]",
    # without positions, a GPT of one block cannot tell "abc" from "bac" at the last place
    _MODEL + "no_positions(model)\nwith torch.no_grad():\n    a = model(torch.tensor([[0, 1, 2]]))[0, -1]\n    b = model(torch.tensor([[1, 0, 2]]))[0, -1]\n"
    "bool(torch.allclose(a, b, atol=1e-5))",
    # a training step does not bring the places back
    _MODEL + "no_positions(model)\nopt = torch.optim.AdamW(model.parameters(), lr=0.1)\n"
    "loss = model(torch.tensor([[0, 1, 2, 3]])).sum()\nloss.backward()\nopt.step()\nround(float(model.pos.weight.abs().sum()), 4)",
    # another GPT: 2 blocks, 6 places
    "import torch\nfrom llm.gpt import GPT\n\ntorch.manual_seed(1)\nmodel = GPT(7, 12, 3, 2, 6)\nno_positions(model)\n"
    "[round(float(model.pos.weight.abs().sum()), 4), model.pos.weight.requires_grad, " + _TRAINABLE + "]",
]

_SCENARIO = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

# The lesson is done when these are correct. "table" is the choice question, "what-it-shows" the open question.
REQUIRED = ["no-positions", "table", "what-it-shows"]

EXERCISES = {
    "no-positions": ScenarioBox(
        name="no-positions",
        start=_START,
        solution=_a(),
        scenarios=_SCENARIOS,
        save_as=("llm", "ablate"),
        first=[_DOTS],
        mistakes=[
            Mistake("in-place", error=lambda e: isinstance(e, RuntimeError) and "leaf Variable that requires grad" in str(e),
                    en="PyTorch stops: the table of places is a parameter, PyTorch records every calculation with it, and so it does not let you change its numbers in place. Put that line under `with torch.no_grad():`, as in the worked example.",
                    nl="PyTorch stopt: de tabel met plekken is een parameter, PyTorch neemt elke berekening ermee op, en daarom mag je zijn getallen niet ter plekke veranderen. Zet die regel onder `with torch.no_grad():`, zoals in het uitgewerkte voorbeeld."),
            Mistake("no-freeze", wrong=_a(freeze=""),
                    en=_SCENARIO + ": the table is 0, but it still trains, so the optimizer learns the places back. Set `model.pos.weight.requires_grad` to `False`: that of the table itself.",
                    nl=_SCENARIO_NL + ": de tabel is 0, maar traint nog, dus de optimizer leert de plekken terug. Zet `model.pos.weight.requires_grad` op `False`: die van de tabel zelf."),
            Mistake("no-zero", wrong=_a(zero=""),
                    en=_SCENARIO + ": the table no longer trains, but it keeps its random numbers, and random rows still tell the places apart. Set every number of it to 0 first.",
                    nl=_SCENARIO_NL + ": de tabel traint niet meer, maar houdt zijn willekeurige getallen, en willekeurige rijen houden de plekken nog steeds uit elkaar. Zet eerst elk getal ervan op 0."),
            Mistake("all-frozen", wrong=_a(freeze="    for p in model.parameters():\n        p.requires_grad = False\n"),
                    en=_SCENARIO + ": nothing of your GPT trains any more. Only the table of places, `model.pos.weight`, should stop training; the rest has to learn.",
                    nl=_SCENARIO_NL + ": er traint niets meer van je GPT. Alleen de tabel met plekken, `model.pos.weight`, moet stoppen met trainen; de rest moet leren."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the first with the line that sets every number of `model.pos.weight` to 0, under `with torch.no_grad():`; the second with the line that stops it from training.",
                      "nl": "Vervang de twee `...`: de eerste door de regel die elk getal van `model.pos.weight` op 0 zet, onder `with torch.no_grad():`; de tweede door de regel die hem laat stoppen met trainen."},
            "wrong": {"en": _SCENARIO + ". Is every number of the table of places 0, does that table, and only that one, no longer train, and does the rest of your GPT stay as it was?",
                      "nl": _SCENARIO_NL + ". Is elk getal van de tabel met plekken 0, traint die tabel, en alleen die, niet meer, en blijft de rest van je GPT zoals hij was?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does it work for every GPT, and after a step of training?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Werkt het voor elke GPT, en na een stap training?"},
            "correct": {"en": "Your `no_positions` is right in all {n} cases: the table of places is 0 and stays 0, and the rest still trains. It is in your project now, as `{saved}`.",
                        "nl": "Je `no_positions` klopt in alle {n} gevallen: de tabel met plekken is 0 en blijft 0, en de rest traint nog. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `no_positions` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `no_positions` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


def last_run() -> dict | None:
    return T.last_run(NAME)


def train(show, seed: int = 0) -> dict | str | None:
    """The run of gpt-train-chars on this computer's plan, with the learner's no_positions applied to the GPT before
    it trains, saved under NAME. None when ablate.py is not in the project yet. Where the learner's gpt_run, gpt_loaders
    or any module they build on is missing or broken, the course's version stands in, and the run's `stand_ins` says so."""
    if not (projects.folder() / "llm" / "ablate.py").exists():
        return None
    from everything_ai.check import run_code
    run = run_code("from llm.ablate import no_positions")
    if run.error is not None:
        return "run"
    return T.train(show, seed=seed, ablate=run.names["no_positions"], name=NAME, own=False)


def control() -> tuple[dict | None, str]:
    """The run with positions to compare with: the learner's own run of gpt-train-chars when it was on this computer's
    plan, all its epochs, else the course's run of that plan with seed 0. And who made it: "yours" or "course"."""
    p = T.plan()
    mine = T.last_run()
    if mine and list(mine.get("sizes", [])) == list(p["sizes"]) and mine.get("epochs") == p["epochs"] and not mine.get("stopped"):
        return mine, "yours"
    plan_name = "cpu" if p["device"] == "cpu" else "gpu"
    ref = T.REFERENCE.get(plan_name) or []
    if not ref:
        return None, "course"
    history = [{"epoch": e, "train_loss": tr, "val_loss": va} for e, tr, va in ref[0]["history"]]
    best = min(history, key=lambda h: h["val_loss"])
    return {"history": history, "best": best["val_loss"], "best_epoch": best["epoch"], "epochs": len(history)}, "course"


def pair_picture(with_pos: list, without: list, labels: dict, epochs: int) -> str:
    """The validation loss after every epoch of the run with positions, dashed and open, and of the run without, solid:
    both a loss, so both purple; the line tells them apart. A grey line at the loss of guessing among 65 characters."""
    import math
    from everything_ai import chart
    n = max(epochs, 2)
    values = [h["val_loss"] for h in with_pos + without]
    low = math.floor(min(values) * 2 - 0.4) / 2
    panel = chart.Panel((0.5, n + 0.5), (low, 4.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line([0.5, n + 0.5], [math.log(T.VOCAB)] * 2, "ink-3", 1, dashed=True)
    panel.text(n + 0.4, math.log(T.VOCAB) + 0.22, labels["guess"], "ink-3", "end")
    for runs, dashed, width in ((with_pos, True, 1.6), (without, False, 2.4)):
        xs = [h["epoch"] for h in runs]
        panel.line(xs, [h["val_loss"] for h in runs], "loss", width, dashed=dashed)
        panel.points(xs, [h["val_loss"] for h in runs], "loss", hollow=dashed)
    panel.text(0.6, 3.75, f"○ {labels['with']}   ● {labels['without']}", "loss", "start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def number(x: float, digits: int, lang: str) -> str:
    return T.number(x, digits, lang)


def table_picture(rows: list, labels: dict) -> str:
    """An ablation table as papers print it: one row per model, the full model first, and its validation loss."""
    head = (f'<tr><th style="text-align:left;padding:0.2rem 0.6rem">{html.escape(labels["model"])}</th>'
            f'<th style="text-align:right;padding:0.2rem 0.6rem">{html.escape(labels["loss"])}</th></tr>')
    body = "".join(f'<tr><td style="padding:0.2rem 0.6rem">{html.escape(name)}</td>'
                   f'<td style="text-align:right;padding:0.2rem 0.6rem;font-family:var(--ea-font-code)">{html.escape(value)}</td></tr>'
                   for name, value in rows)
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink)">'
            f'<table style="border-collapse:collapse">{head}{body}</table></figure>')


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("no-positions", _START, "empty", None),
    ("no-positions", _a(), "correct", None),
    ("no-positions", _a(zero="    with torch.no_grad():\n        model.pos.weight.fill_(0.0)\n"), "correct", None),
    ("no-positions", _a(zero="    with torch.no_grad():\n        model.pos.weight.zero_()\n", freeze="    model.pos.weight.requires_grad_(False)\n"), "correct", None),
    ("no-positions", _a(zero="    torch.nn.init.zeros_(model.pos.weight)\n"), "correct", None),
    ("no-positions", _a(zero="    model.pos.weight[:] = 0\n"), "error", "in-place"),
    ("no-positions", _a(zero="    with torch.no_grad():\n        model.pos.weight[0] = 0\n"), "wrong", None),
    ("no-positions", _a(zero="    with torch.no_grad():\n        model.pos.weight[:] = 0\n        model.tok.weight[:] = 0\n"), "wrong", None),
    ("no-positions", _a(freeze=""), "wrong", "no-freeze"),
    ("no-positions", _a(zero=""), "wrong", "no-zero"),
    ("no-positions", _a(freeze="    for p in model.parameters():\n        p.requires_grad = False\n"), "wrong", "all-frozen"),
    ("no-positions", _START.replace("    ...\n", _ZERO, 1), "wrong", "dots-left"),
]


def tiny_gpt():
    """A small GPT of gpt-assemble for the worked example, with seed 0: 5 characters, d = 8, 2 heads, 1 block, 4
    places. The learner's GPT, or the course's where theirs is missing or broken."""
    import torch
    from everything_ai.check import run_code
    run = run_code("from llm.gpt import GPT")
    torch.manual_seed(0)
    return run.names["GPT"](5, 8, 2, 1, 4)


def flow_picture(labels: dict) -> str:
    """The parts of the GPT from input to output, the table of places struck through: ink only, since the picture is
    about which part is gone, not about a role. HTML boxes, so the text wraps in a narrow window."""
    def box(text_, struck=False, note=""):
        # only the name is struck through: the note says what the table holds now, and stays readable
        inner = f'<span style="text-decoration:line-through">{html.escape(text_)}</span>' if struck else html.escape(text_)
        inner += f'<span style="color:var(--ea-ink-2)">: {html.escape(note)}</span>' if note else ""
        line = "border-style:dashed;color:var(--ea-ink-3)" if struck else "color:var(--ea-ink)"
        return (f'<span style="display:inline-block;padding:0.3rem 0.6rem;border:1.2px solid var(--ea-ink-3);border-radius:6px;'
                f'background:var(--ea-inset);{line}">{inner}</span>')
    arrow = '<div style="text-align:center;color:var(--ea-ink-3);margin:0.2rem 0">↓</div>'
    rows = [f'<div style="text-align:center">{box(labels["tok"])} <span style="color:var(--ea-ink-2)">{html.escape(labels["plus"])}</span> {box(labels["pos"], True, labels["pos_note"])}</div>',
            f'<div style="text-align:center">{box(labels["blocks"])}</div>',
            f'<div style="text-align:center">{box(labels["head"])}</div>',
            f'<div style="text-align:center;color:var(--ea-ink-2)">{html.escape(labels["out"])}</div>']
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:560px;font-family:var(--ea-font-ui);font-size:0.875rem;line-height:1.6">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'{arrow.join(rows)}</figure>')
