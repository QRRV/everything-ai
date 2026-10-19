# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-compute-6nd: train_hours, the time of a training run from the 6ND rule, the peak of a GPU,
the part of it a run reaches and the number of GPUs. And the demo: the learner's train_hours on runs from their own
GPT on a book to a model of 70 billion parameters."""
import math
import re
from decimal import Decimal

from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit


def token_picture(labels: dict) -> str:
    """What training on one token costs, in units of N FLOPs: the forward pass 2N, the backward pass 4N, and attention
    on top, about n/(12d) of 6N: 5.6% for the GPT of the book (n = 256, d = 384), 1.4% for GPT-3 (n = 2,048,
    d = 12,288). FLOPs are counted work, not a role of the model, so the bars are ink; attention in the strongest ink."""
    from everything_ai import chart
    p = chart.Panel((0, 6.7), (-0.25, 2.65), title=labels["title"], xlabel=labels["x"], xticks=[0, 2, 4, 6], yticks=[])
    for row, (name, n, d) in enumerate(((labels["gpt3"], 2048, 12288), (labels["book"], 256, 384))):
        y = row * 1.35
        attention = 6 * n / (12 * d)
        p.rect(0, y, 2, y + 0.7, "ink-3")
        p.rect(2, y, 6, y + 0.7, "ink-2")
        p.rect(6, y, 6 + attention, y + 0.7, "ink")
        p.text(1, y + 0.35, labels["forward"], "ink", "middle")
        p.text(4, y + 0.35, labels["backward"], "ink", "middle")
        p.text(0, y + 0.95, labels["row"].format(name=name, pct=number(100 * attention / 6, 1, labels["lang"])), "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels["lang"] == "nl" else svg


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


UNITS = {"en": ("seconds", "minutes", "hours", "days"), "nl": ("seconden", "minuten", "uur", "dagen")}


def duration(hours: float, lang: str) -> str:
    """A time in hours in the unit that suits it: seconds, minutes, hours up to two days, then days."""
    s, m, h, d = UNITS[lang]
    if hours < 1 / 60:
        return f"{number(hours * 3600, 0, lang)} {s}"
    if hours < 1:
        return f"{number(hours * 60, 0, lang)} {m}"
    if hours < 48:
        return f"{number(hours, 1, lang)} {h}"
    return f"{number(hours / 24, 0, lang)} {d}"


# --- the demo: the learner's train_hours from a book to 70 billion parameters --------------------------------------------------

# (key, parameters, tokens, peak FLOP/s, part of the peak, GPUs): an A100 in BF16 at 40% of its peak for every run.
RUNS = [("book", 10.8e6, 20e6, 312e12, 0.4, 1),
        ("small", 124e6, 10e9, 312e12, 0.4, 8),
        ("7b", 7e9, 2e12, 312e12, 0.4, 1024),
        ("70b", 70e9, 2e12, 312e12, 0.4, 2048)]


def demo(code: str):
    """The learner's train_hours on every run of RUNS: a list of (key, hours, GPU hours), or "error" when their code
    stops with an error or takes too long."""
    yours = run_code(code)
    if yours.error is not None or "train_hours" not in yours.names:
        return "error"
    train_hours = yours.names["train_hours"]

    def work():
        return [(key, float(train_hours(n, d, peak, part, gpus)), float(train_hours(n, d, peak, part, gpus)) * gpus)
                for key, n, d, peak, part, gpus in RUNS]

    try:
        return with_time_limit(work, 10)
    except Exception:
        return "error"


# --- exercise 1: train_hours ----------------------------------------------------------------------------------------------------

_HEAD = "def train_hours(n_params, n_tokens, peak, utilization, n_gpus):\n"
_START = (_HEAD + "    # the FLOPs of the whole run, and the FLOPs per second that all GPUs together really do\n    ...\n")


def _t(rule: str = "6 * n_params * n_tokens", speed: str = "peak * utilization * n_gpus", unit: str = " / 3600") -> str:
    return _HEAD + f"    flops = {rule}\n    speed = {speed}\n    return flops / speed{unit}\n"


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _show(v) -> str:
    """A value for the messages: a number to 6 digits and never in e-notation, 0.00000000732593 and not 7.32593e-09;
    whole from 100,000 on, and past 10**15 6 digits and then zeros, not the float's noise. A list or tuple number by
    number; anything else, such as text, as Python writes it."""
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_show(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    if isinstance(v, (str, bool)):
        return repr(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if not math.isfinite(x):
        return repr(x)
    if 1e5 <= abs(x) < 1e15:
        return f"{x:.0f}"
    return format(Decimal(f"{x:.6g}"), "f")


class _Num(float):
    """A number that a check message writes the way the lesson does, such as `989e12`."""

    def __new__(cls, value, shown):
        obj = super().__new__(cls, value)
        obj.shown = shown
        return obj

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return float(self)


def _n(text: str) -> _Num:
    return _Num(float(text), text)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "train-hours": FunctionBox(
        name="train-hours",
        function="train_hours",
        start=_START,
        solution=_t(),
        visible=[(_n("1e9"), _n("20e9"), _n("989e12"), 0.4, 8)],
        hidden=[(_n("124e6"), _n("10e9"), _n("312e12"), 0.5, 1), (_n("10.8e6"), _n("20e6"), _n("82.6e12"), 0.3, 1),
                (_n("70e9"), _n("2e12"), _n("312e12"), 0.4, 2048), (_n("7e9"), _n("1e12"), _n("989e12"), 0.35, 64)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("forward-only", wrong=_t(rule="2 * n_params * n_tokens"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: three times too little. You count 2 FLOPs per parameter per token, the forward pass; training adds the backward pass, which costs about twice that.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: drie keer te weinig. Je telt 2 FLOPs per parameter per token, de forward pass; trainen voegt de backward pass toe, die ongeveer twee keer zoveel kost."),
            Mistake("no-utilization", wrong=_t(speed="peak * n_gpus"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your GPUs run at their full peak. A run reaches only `utilization` of it, so it does fewer FLOPs a second and takes longer.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je GPU's draaien op hun volle piek. Een run haalt daar alleen `utilization` van, dus doet hij minder FLOPs per seconde en duurt hij langer."),
            Mistake("divided-utilization", wrong=_t(speed="peak / utilization * n_gpus"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide the peak by `utilization`, which makes the GPUs faster than their peak. They do `peak * utilization` FLOPs a second.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt de piek door `utilization`, waardoor de GPU's sneller worden dan hun piek. Ze doen `peak * utilization` FLOPs per seconde."),
            Mistake("no-gpus", wrong=_t(speed="peak * utilization"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the time on one GPU. With `n_gpus` GPUs, they do `n_gpus` times as many FLOPs a second together.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de tijd op één GPU. Met `n_gpus` GPU's doen ze samen `n_gpus` keer zoveel FLOPs per seconde."),
            Mistake("seconds", wrong=_t(unit=""),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in seconds, as the peak is in FLOPs a second. An hour has 3,600 seconds.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in seconden, want de piek is in FLOPs per seconde. Een uur heeft 3.600 seconden."),
            Mistake("minutes", wrong=_t(unit=" / 60"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in minutes. An hour has 3,600 seconds.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in minuten. Een uur heeft 3.600 seconden."),
            Mistake("days", wrong=_t(unit=" / 3600 / 24"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in days. Give the time in hours.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in dagen. Geef de tijd in uren."),
            Mistake("percent", wrong=_t(speed="peak * utilization / 100 * n_gpus"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: 100 times too long. `utilization` is already a part, such as `0.4` for 40%; do not divide it by 100.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: 100 keer te lang. `utilization` is al een deel, zoals `0.4` voor 40%; deel het niet door 100."),
            Mistake("rounded", matches=lambda got, text: re.search(r"^[^#\n]*(\bround\(|\bint\(|//)", text, re.MULTILINE) is not None,
                    en="Your function rounds, with `round`, `int` or `//`, and the check compares the time of `{call}` to many digits. Divide with `/`, and give the time without rounding.",
                    nl="Je functie rondt af, met `round`, `int` of `//`, en de check vergelijkt de tijd van `{call}` op veel cijfers. Deel met `/`, en geef de tijd zonder af te ronden."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: the FLOPs of the run, the FLOPs per second of all GPUs together, and a `return` of the time in hours.",
                      "nl": "Vervang de `...` door je code: de FLOPs van de run, de FLOPs per seconde van alle GPU's samen, en een `return` van de tijd in uren."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the parameters, the tokens, the peak, the part of the peak and the number of GPUs, and the `def` line of your function has another number of names between its parentheses. Keep `(n_params, n_tokens, peak, utilization, n_gpus)`.",
                          "nl": "De check roept `{call}` aan, met de parameters, de tokens, de piek, het deel van de piek en het aantal GPU's, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n_params, n_tokens, peak, utilization, n_gpus)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you count the FLOPs of the whole run with the 6ND rule, divide them by what all GPUs together really do in a second, and give hours?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je de FLOPs van de hele run met de 6ND-vuistregel, deel je ze door wat alle GPU's samen echt doen in een seconde, en geef je uren?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use all five numbers it gets, and no fixed ones?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie alle vijf getallen die hij krijgt, en geen vaste?"},
            "correct": {"en": "Your function is right in all {n} cases, from a GPT of 10.8 million parameters to one of 70 billion on 2,048 GPUs.",
                        "nl": "Je functie klopt in alle {n} gevallen, van een GPT van 10,8 miljoen parameters tot een van 70 miljard op 2.048 GPU's."},
        },
    ),
}

# The lesson is done when these are correct. "budget" is the choice question, "too-optimistic" the open question.
REQUIRED = ["train-hours", "budget", "too-optimistic"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("train-hours", _START, "empty", None),
    ("train-hours", _t(), "correct", None),
    ("train-hours", _HEAD + "    return 6 * n_params * n_tokens / (peak * utilization * n_gpus * 3600)\n", "correct", None),
    ("train-hours", _HEAD + "    seconds = 6 * n_params * n_tokens / peak / utilization / n_gpus\n    return seconds / 60 / 60\n", "correct", None),
    ("train-hours", _t(rule="2 * n_params * n_tokens"), "wrong", "forward-only"),
    ("train-hours", _t(speed="peak * n_gpus"), "wrong", "no-utilization"),
    ("train-hours", _t(speed="peak / utilization * n_gpus"), "wrong", "divided-utilization"),
    ("train-hours", _t(speed="peak * utilization"), "wrong", "no-gpus"),
    ("train-hours", _t(unit=""), "wrong", "seconds"),
    ("train-hours", _t(unit=" / 60"), "wrong", "minutes"),
    ("train-hours", _t(unit=" / 3600 / 24"), "wrong", "days"),
    ("train-hours", _t(speed="peak * 0.4 * n_gpus"), "almost", None),
    ("train-hours", _t(speed="peak * utilization / 100 * n_gpus"), "wrong", "percent"),
    ("train-hours", _HEAD + "    return round(6 * n_params * n_tokens / (peak * utilization * n_gpus) / 3600, 2)\n", "wrong", "rounded"),
    ("train-hours", _HEAD + "    return 6 * n_params * n_tokens // (peak * utilization * n_gpus) / 3600\n", "wrong", "rounded"),
    ("train-hours", _HEAD + "    return int(6 * n_params * n_tokens / (peak * utilization * n_gpus) / 3600)\n", "wrong", "rounded"),
    ("train-hours", _HEAD + "    return (peak * utilization * n_gpus) / (6 * n_params * n_tokens) / 3600\n", "wrong", None),
]
