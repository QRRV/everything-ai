# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-hypothesis-control: a variant that differs from the control run in one setting. And the
experiment of the demo: the control run and one variant of a small GPT, with the learner's GPT, gpt_loaders and
Trainer, on the first quarter of Tiny Shakespeare."""
import html
import math
import re
import time

import torch

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code

# The runs use the learner's GPT of gpt-assemble, with their Block and attention, their gpt_examples, their Trainer of
# nn-trainer and their gpt_loaders of gpt-train-chars; where one is missing or broken, the course's version stands in.
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"),
                   ("llm", "gpt_data", "gpt-assemble", "gpt-examples"),
                   ("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("llm", "gpt_loaders", "gpt-train-chars", "gpt-loaders"))

DATA = "data/tiny-shakespeare.txt"
PART = 0.25   # the runs train on the first quarter of the book, so that one run takes well under a minute on a CPU

# The control run of the lesson: a small GPT, small enough to train twice in a minute or two on any computer.
CONTROL = {"lr": 0.001, "warmup": 100, "layers": 2, "d": 64, "heads": 4, "block": 64, "batch": 32, "epochs": 6, "seed": 0}
# The changes the demo offers, each one setting: (name, value).
CHANGES = [("lr", 0.003), ("warmup", 0), ("layers", 4)]

# The course's own runs of these experiments, seeds 0, 1 and 2, on a CPU with 1 thread: the validation loss after the
# last epoch. "control" and "variant": the lowest and the highest of the 3 seeds. "lower": by how much the variant ended
# below the control with the same seed, the smallest and the largest of the 3, and the digits to show them with; every
# variant ended lower with every seed. With seeds 0, 1, 2 the control ended at 2.3902, 2.3924, 2.4174.
REFERENCE = {
    "lr=0.003": {"control": (2.390, 2.417), "variant": (2.083, 2.099), "lower": (0.29, 0.33, 2)},    # 2.0827, 2.0987, 2.0850
    "warmup=0": {"control": (2.390, 2.417), "variant": (2.384, 2.412), "lower": (0.004, 0.008, 3)},  # 2.3866, 2.3841, 2.4115
    "layers=4": {"control": (2.390, 2.417), "variant": (2.325, 2.337), "lower": (0.06, 0.08, 2)},    # 2.3335, 2.3245, 2.3374
}


class _Named(dict):
    """A dictionary that a check message calls by its name, `control`, instead of writing out all its pairs."""

    def __init__(self, name: str, data: dict):
        super().__init__(data)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        # the learner's function gets a plain dictionary, so a copy it makes, also with copy.copy, is a plain one too
        return dict(self)


def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def changed(a: dict, b: dict) -> list:
    """The names of the settings in which two runs differ."""
    return [name for name in a if a[name] != b.get(name)]


class _Counted:
    """The training mini-batches of a run, counted as they go, so the demo can show how far it is."""

    def __init__(self, loader, step):
        self.loader = loader
        self.step = step

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.step()
            yield batch


def run(settings: dict, show=None) -> list | str:
    """Train one run of the small GPT with these settings, on the CPU, and give back its history: a dict per epoch
    with epoch, train_loss and val_loss. "run" when the learner's code stops with an error, "stopped" after the stop
    button of marimo (a KeyboardInterrupt). `show` gets how far the run is, as a share from 0 to 1, about every second."""
    imports = run_code("from llm.gpt import GPT\nfrom llm.gpt_loaders import gpt_loaders\nfrom nn.trainer import Trainer")
    if imports.error is not None:
        return "run"
    GPT, gpt_loaders, Trainer = (imports.names[k] for k in ("GPT", "gpt_loaders", "Trainer"))
    book = text()
    chars = sorted(set(book))   # all 65 characters of the book, so the model reads them as in gpt-train-chars
    index = {ch: i for i, ch in enumerate(chars)}
    ids = [index[ch] for ch in book[:int(PART * len(book))]]
    s = settings
    try:
        torch.manual_seed(s["seed"])
        train_loader, val_loader = gpt_loaders(ids, s["block"], s["batch"], "cpu")
        model = GPT(len(chars), s["d"], s["heads"], s["layers"], s["block"])
        trainer = Trainer(model, train_loader, val_loader, s["epochs"], lr=s["lr"], warmup=s["warmup"], max_norm=1.0)
        steps = s["epochs"] * len(train_loader)
        state = {"step": 0, "shown": 0.0}

        def step():
            state["step"] += 1
            now = time.time()
            if show and now - state["shown"] >= 1:
                state["shown"] = now
                show(state["step"] / steps)

        trainer.train_loader = _Counted(trainer.train_loader, step)
        history = trainer.fit()
    except KeyboardInterrupt:   # the stop button of marimo
        return "stopped"
    except Exception:
        return "run"
    return [{"epoch": int(h["epoch"]), "train_loss": float(h["train_loss"]), "val_loss": float(h["val_loss"])} for h in history]


def settings_picture(control: dict, variant: dict, labels: dict) -> str:
    """The settings of the control run and of the variant side by side, the one that differs marked in bold with an
    arrow. Ink only: every row is a setting, so a colour would not say which one changed. HTML, so it fits a phone."""
    rows = []
    for name in control:
        diff = control[name] != variant.get(name)
        weight = "font-weight:700;color:var(--ea-ink)" if diff else "color:var(--ea-ink-2)"
        mark = f'← {html.escape(labels["changed"])}' if diff else ""
        rows.append(f'<tr><td style="padding:0.15rem 0.6rem;font-family:var(--ea-font-code);{weight}">{html.escape(name)}</td>'
                    f'<td style="padding:0.15rem 0.6rem;text-align:right;font-family:var(--ea-font-code);{weight}">{html.escape(str(control[name]))}</td>'
                    f'<td style="padding:0.15rem 0.6rem;text-align:right;font-family:var(--ea-font-code);{weight}">{html.escape(str(variant.get(name)))}</td>'
                    f'<td style="padding:0.15rem 0.3rem;color:var(--ea-ink)">{mark}</td></tr>')
    head = (f'<tr><th style="padding:0.15rem 0.6rem;text-align:left">{html.escape(labels["setting"])}</th>'
            f'<th style="padding:0.15rem 0.6rem;text-align:right">{html.escape(labels["control"])}</th>'
            f'<th style="padding:0.15rem 0.6rem;text-align:right">{html.escape(labels["variant"])}</th><th></th></tr>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;font-family:var(--ea-font-ui);font-size:0.875rem;color:var(--ea-ink-2)">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.4rem">{html.escape(labels["title"])}</div>'
            f'<table style="border-collapse:collapse">{head}{"".join(rows)}</table></figure>')


def curves_picture(control: list, variant: list, labels: dict) -> str:
    """The validation loss after every epoch of the control run, dashed and open, and of the variant, solid: both a loss,
    so both purple; the line tells them apart."""
    from everything_ai import chart
    n = max(len(control), len(variant), 2)
    values = [h["val_loss"] for h in control + variant]
    low, high = math.floor(min(values) * 5) / 5 - 0.1, math.ceil(max(values) * 5) / 5 + 0.1
    panel = chart.Panel((0.5, n + 0.5), (low, high), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for runs, dashed, width in ((control, True, 1.6), (variant, False, 2.4)):
        xs = [h["epoch"] for h in runs]
        panel.line(xs, [h["val_loss"] for h in runs], "loss", width, dashed=dashed)
        panel.points(xs, [h["val_loss"] for h in runs], "loss", hollow=dashed)
    panel.text(n + 0.4, high - (high - low) * 0.08, f"○ {labels['control']}   ● {labels['variant']}", "loss", "end")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:.{digits}f}"
    return text_.replace(".", ",") if lang == "nl" else text_


def _changes_control(got, text) -> bool:
    """The learner's one_change changes the dictionary it gets: the variant is the control itself, under a new name."""
    ran = run_code(text)
    fn = ran.names.get("one_change") if ran.error is None else None
    if not callable(fn):
        return False
    control = dict(CONTROL)   # the call the message names
    try:
        fn(control, "lr", 0.003)
    except Exception:
        return False
    return control != CONTROL


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_SAME = Mistake("same-dict", matches=_changes_control,
                en="Your function changes the dictionary it gets: after `one_change(control, \"lr\", 0.003)`, `control` itself is changed too, and your control run is gone. `variant = control` gives the same dictionary a second name; `dict(control)` makes a new one with the same pairs.",
                nl="Je functie verandert de dictionary die hij krijgt: na `one_change(control, \"lr\", 0.003)` is `control` zelf ook veranderd, en is je control run weg. `variant = control` geeft dezelfde dictionary een tweede naam; `dict(control)` maakt een nieuwe met dezelfde paren.")

_START = ("def one_change(control, name, value):\n"
          "    # a new dictionary with the same pairs as control\n"
          "    variant = ...\n"
          "    # the one setting that differs\n"
          "    ...\n"
          "    return variant\n")
_SOLUTION = ("def one_change(control, name, value):\n"
             "    variant = dict(control)\n"
             "    variant[name] = value\n"
             "    return variant\n")


def one_change_of(text: str | None):
    """The learner's one_change from the code of their box, when it is right, else the course's: (function, theirs)."""
    if text:
        ran = run_code(text)
        fn = ran.names.get("one_change") if ran.error is None else None
        if callable(fn):
            return fn, True
    return run_code(_SOLUTION).names["one_change"], False

# The lesson is done when these are correct. "confound" is the choice question, "design" the open question.
REQUIRED = ["one-change", "confound", "design"]

EXERCISES = {
    "one-change": FunctionBox(
        name="one-change",
        function="one_change",
        start=_START,
        solution=_SOLUTION,
        visible=[(_Named("control", CONTROL), "warmup", 0)],
        hidden=[(_Named("control", CONTROL), "layers", 4), (_Named("control", CONTROL), "lr", 0.003), ({"a": 1, "b": 2}, "b", 3)],
        first=[_DOTS, _SAME],
        mistakes=[
            Mistake("only-pair", wrong=_SOLUTION.replace("    variant = dict(control)\n", "    variant = {}\n"),
                    en="`{call}` gives `{got}`: only the one setting. The variant needs every setting of the control, so make it from `control`, and then change the one.",
                    nl="`{call}` geeft `{got}`: alleen die ene instelling. De variant heeft elke instelling van de control nodig, dus maak hem van `control`, en verander dan die ene."),
            Mistake("quoted-name", wrong=_SOLUTION.replace("variant[name] = value", "variant[\"name\"] = value"),
                    en="`{call}` gives `{got}`, with a new key called `\"name\"`. Use the name `name` without quotes: it holds the setting the check asks for, such as `\"warmup\"`.",
                    nl="`{call}` geeft `{got}`, met een nieuwe key die `\"name\"` heet. Gebruik de naam `name` zonder aanhalingstekens: daarin zit de instelling waarnaar de check vraagt, zoals `\"warmup\"`."),
            Mistake("unchanged", wrong=_SOLUTION.replace("    variant[name] = value\n", ""),
                    en="`{call}` gives the control again, with nothing changed. Does your function give back the variant, with the setting `name` changed to `value`?",
                    nl="`{call}` geeft de control opnieuw, zonder verandering. Geeft je functie de variant terug, met de instelling `name` veranderd in `value`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the first with a new dictionary with the pairs of `control`, the second with the line that gives the setting `name` the value `value`.",
                      "nl": "Vervang de twee `...`: de eerste door een nieuwe dictionary met de paren van `control`, de tweede door de regel die de instelling `name` de waarde `value` geeft."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the control, a name and a value, and the `def` line of your function has a different number of names between its parentheses. Keep `(control, name, value)`.",
                          "nl": "De check roept `{call}` aan, met de control, een naam en een waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(control, name, value)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does the variant have every pair of `control`, with only the setting `name` changed to `value`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Heeft de variant elk paar van `control`, met alleen de instelling `name` veranderd in `value`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every setting, and for another dictionary?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke instelling, en voor een andere dictionary?"},
            "correct": {"en": "Your function is right in all {n} cases, and leaves the control as it was. The demo below makes its variant with it.",
                        "nl": "Je functie klopt in alle {n} gevallen, en laat de control zoals hij was. De demo hieronder maakt zijn variant ermee."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_D = "def one_change(control, name, value):\n"
CASES = [
    ("one-change", _START, "empty", None),
    ("one-change", _SOLUTION, "correct", None),
    ("one-change", _D + "    variant = control.copy()\n    variant[name] = value\n    return variant\n", "correct", None),
    ("one-change", _D + "    variant = {}\n    for key in control:\n        variant[key] = control[key]\n    variant[name] = value\n    return variant\n", "correct", None),
    ("one-change", _D + "    variant = control\n    variant[name] = value\n    return variant\n", "wrong", "same-dict"),
    ("one-change", _D + "    control[name] = value\n    return control\n", "wrong", "same-dict"),
    ("one-change", _D + "    variant = control\n    variant[\"name\"] = value\n    return variant\n", "wrong", "same-dict"),
    ("one-change", "import copy\n\n\n" + _D + "    variant = copy.copy(control)\n    variant[name] = value\n    return variant\n", "correct", None),
    ("one-change", _D + "    return {**control, name: value}\n", "correct", None),
    ("one-change", _D + "    variant = dict(control)\n    variant[name] = value\n    return control\n", "wrong", "unchanged"),
    ("one-change", _D + "    variant = {}\n    variant[name] = value\n    return variant\n", "wrong", "only-pair"),
    ("one-change", _SOLUTION.replace("variant[name] = value", "variant[\"name\"] = value"), "wrong", "quoted-name"),
    ("one-change", _D + "    variant = dict(control)\n    return variant\n", "wrong", "unchanged"),
    ("one-change", _START.replace("    variant = ...\n", "    variant = dict(control)\n"), "wrong", "dots-left"),
]
