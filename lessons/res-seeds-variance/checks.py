# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-seeds-variance: seed_summary, the mean of the runs of several seeds and a bootstrap
interval around it. And the demo: the small GPT of res-hypothesis-control, trained with five seeds."""
import importlib.util
import math
import re

import numpy as np

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, _show_rounded


def _hypothesis_control():
    """The checks of res-hypothesis-control, whose small GPT and runs this lesson repeats with more seeds. Loading
    them registers that lesson's project modules, the same as this lesson's."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_hypothesis_control_for_seeds",
                                                  paths.LESSONS / "res-hypothesis-control" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _hypothesis_control()
CONTROL = H.CONTROL
SEEDS = [0, 1, 2, 3, 4]

# The course's runs of the small GPT with seeds 0 to 4, on a CPU with 1 thread: the validation loss after the last
# epoch, of the control run and of the variant without warmup. Filled in from the reference runs.
REFERENCE = {"control": [2.3902, 2.3924, 2.4174, 2.3938, 2.4012], "warmup=0": [2.3866, 2.3841, 2.4115, 2.3916, 2.3962]}


def run_seeds(show=None, hear=None) -> list | str:
    """The control run of res-hypothesis-control with each seed of SEEDS, with the learner's modules: the validation
    loss after the last epoch of each. "run" when a run stops with an error, "stopped" when the stop button ended it.
    `show` gets (which seed, share of that run done) about every second; `hear` gets (which seed, what live.follow
    heard), for a live chart."""
    out = []
    for i, seed in enumerate(SEEDS):
        settings = dict(CONTROL)
        settings["seed"] = seed
        h = H.run(settings, None if show is None else (lambda share, i=i: show(i, share)),
                  None if hear is None else (lambda heard, i=i: hear(i, heard)))
        if isinstance(h, str):
            return h
        out.append(h[-1]["val_loss"])
    return out


def bootstrap_interval(values, n_boot: int = 10000, seed: int = 0) -> tuple:
    """The course's own seed_summary, for the demo: the mean and the 95% bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values)).mean() for _ in range(n_boot)]
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def strip_picture(groups: list, labels: dict) -> str:
    """Each group of runs as a row of points, one per seed, with its mean as a tick and its 95% bootstrap interval as a
    bar: the points are losses, so purple; the interval is ink."""
    from everything_ai import chart
    values = [v for _, vs in groups for v in vs]
    low, high = min(values), max(values)
    pad = (high - low) * 0.25 or 0.01
    panel = chart.Panel((low - pad, high + pad), (0.4, len(groups) + 0.6), title=labels["title"], xlabel=labels["x"], ylabel="", yticks=[])
    for k, (name, vs) in enumerate(groups):
        y = len(groups) - k
        mean, lo, hi = bootstrap_interval(vs)
        panel.line([lo, hi], [y - 0.22, y - 0.22], "ink", 2.4)
        panel.line([mean, mean], [y - 0.32, y - 0.12], "ink", 2.4)
        panel.points(vs, [y] * len(vs), "loss")
        panel.text(low - pad * 0.9, y + 0.18, name, "ink-2", "start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=110 + 70 * len(groups))
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# --- exercise 1: seed_summary ----------------------------------------------------------------------------------------------

_HEAD = "import numpy as np\n\n\ndef seed_summary(losses, n_boot, seed):\n"
_START = (_HEAD + "    # the means of n_boot resamples of losses, each drawn with replacement\n    rng = np.random.default_rng(seed)\n"
          "    means = []\n    ...\n    # the middle 95% of those means\n    low, high = ...\n    return [np.mean(losses), low, high]\n")


def _s(draw: str = "rng.choice(losses, size=len(losses))", cut: str = "np.quantile(means, [0.025, 0.975])",
       ret: str = "[np.mean(losses), low, high]") -> str:
    return (_HEAD + "    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n"
            f"        means.append(np.mean({draw}))\n    low, high = {cut}\n    return {ret}\n")


def _three(got) -> list | None:
    """Three numbers, also when they come as the mean and a pair: [mean, [low, high]] is the same answer."""
    try:
        parts = got if isinstance(got, (list, tuple)) else [got]
        a = np.concatenate([np.asarray(x, dtype=float).ravel() for x in parts]) if parts else np.zeros(0)
    except (TypeError, ValueError):
        return None
    return a.tolist() if a.shape == (3,) and np.all(np.isfinite(a)) else None


def _like(got, want) -> bool:
    """The same mean, and ends of the interval within what another way of drawing gives: a bootstrap is random."""
    g, w = _three(got), _three(want)
    if g is None or w is None:
        return False
    width = max(w[2] - w[1], 1e-9)
    return abs(g[0] - w[0]) < 1e-9 and abs(g[1] - w[1]) < 0.08 * width + 1e-9 and abs(g[2] - w[2]) < 0.08 * width + 1e-9


def _show(v) -> str:
    t = _three(v)
    return "[" + ", ".join(f"{x:.4f}" for x in t) + "]" if t else _show_rounded(v, 4)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# The lesson is done when these are correct. "overlap" is the choice question, "report" the open question.
REQUIRED = ["seed-summary", "overlap", "report"]

EXERCISES = {
    "seed-summary": FunctionBox(
        name="seed-summary",
        function="seed_summary",
        start=_START,
        solution=_s(),
        visible=[([2.41, 2.38, 2.44, 2.40, 2.39], 10000, 0)],
        # the 6 and the 10 runs have uneven values, so their means lie close together: there a 98% interval or a resample
        # of one run too few falls outside the margin of _like, which another way of drawing stays well inside
        hidden=[([1.66, 1.67, 1.66], 10000, 1), ([2.031, 2.214, 2.127, 2.518, 1.932, 2.306], 10000, 2), ([0.5, 0.9], 10000, 3),
                ([1.412, 1.437, 1.398, 1.455, 1.421, 1.409, 1.446, 1.43, 1.393, 1.418], 10000, 4)],
        show=_show,
        compare=_like,
        first=[_DOTS],
        mistakes=[
            Mistake("quantiles-of-runs", wrong=_s(cut="np.quantile(losses, [0.025, 0.975])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your interval runs about from the lowest run to the highest, so it shows how far single runs spread, not how sure their mean is. The interval of the mean comes from the means of whole resamples: take the quantiles of `means`, and let each resample draw as many runs as there are, `size=len(losses)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je interval loopt ongeveer van de laagste run tot de hoogste, dus het laat zien hoe ver losse runs spreiden, niet hoe zeker hun gemiddelde is. Het interval van het gemiddelde komt uit de gemiddelden van hele resamples: neem de kwantielen van `means`, en laat elke resample evenveel runs trekken als er zijn, `size=len(losses)`."),
            Mistake("ninety", wrong=_s(cut="np.quantile(means, [0.05, 0.95])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your interval holds the middle 90% of the means. For 95%, cut 2.5% off each side: `[0.025, 0.975]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je interval bevat de middelste 90% van de gemiddelden. Voor 95% haal je aan elke kant 2,5% eraf: `[0.025, 0.975]`."),
            Mistake("ninety-eight", wrong=_s(cut="np.quantile(means, [0.01, 0.99])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your interval holds the middle 98% of the means. For 95%, cut 2.5% off each side: `[0.025, 0.975]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je interval bevat de middelste 98% van de gemiddelden. Voor 95% haal je aan elke kant 2,5% eraf: `[0.025, 0.975]`."),
            Mistake("same-resample", wrong=_HEAD + "    means = []\n    for _ in range(n_boot):\n        rng = np.random.default_rng(seed)\n"
                    "        means.append(np.mean(rng.choice(losses, size=len(losses))))\n    low, high = np.quantile(means, [0.025, 0.975])\n"
                    "    return [np.mean(losses), low, high]\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: every resample is the same, so the interval has no width. Draw a new resample in every round of the loop, with the one `rng` you made before it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke resample is hetzelfde, dus het interval heeft geen breedte. Trek in elke ronde van de loop een nieuwe resample, met de ene `rng` die je ervoor maakte."),
            Mistake("no-replacement", wrong=_s(draw="rng.permutation(losses)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every resample holds each run exactly once, so every mean is the same. Draw with replacement: `rng.choice(losses, size=len(losses))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke resample bevat elke run precies één keer, dus elk gemiddelde is hetzelfde. Trek met teruglegging: `rng.choice(losses, size=len(losses))`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the first with a loop that adds the mean of a resample of `losses` to `means`, `n_boot` times; the second with the two quantiles of `means` that hold the middle 95%.",
                      "nl": "Vervang de twee `...`: de eerste door een loop die `n_boot` keer het gemiddelde van een resample van `losses` aan `means` toevoegt; de tweede door de twee kwantielen van `means` die de middelste 95% bevatten."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the losses, the number of resamples and a seed, and the `def` line of your function has another number of names between its parentheses. Keep `(losses, n_boot, seed)`.",
                          "nl": "De check roept `{call}` aan, met de losses, het aantal resamples en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(losses, n_boot, seed)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take `n_boot` means of resamples drawn with replacement, and the quantiles of those means at 0.025 and 0.975?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je `n_boot` gemiddelden van resamples die met teruglegging getrokken zijn, en de kwantielen van die gemiddelden bij 0,025 en 0,975?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of runs, and does it cut exactly 2.5% off each side?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het voor elk aantal runs, en haalt het aan elke kant precies 2,5% eraf?"},
            "correct": {"en": "Your function is right in all {n} cases: the mean of the runs, and the interval in which the mean would mostly land with other seeds.",
                        "nl": "Je functie klopt in alle {n} gevallen: het gemiddelde van de runs, en het interval waarin het gemiddelde met andere seeds meestal zou landen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_D = _HEAD
CASES = [
    ("seed-summary", _START, "empty", None),
    ("seed-summary", _s(), "correct", None),
    ("seed-summary", _s(draw="np.array(losses)[rng.integers(0, len(losses), size=len(losses))]"), "correct", None),
    ("seed-summary", _s(ret="(np.mean(losses), low, high)"), "correct", None),
    ("seed-summary", _s(cut="np.quantile(losses, [0.025, 0.975])"), "wrong", "quantiles-of-runs"),
    ("seed-summary", _s(cut="np.quantile(means, [0.05, 0.95])"), "wrong", "ninety"),
    ("seed-summary", _s(cut="min(losses), max(losses)"), "wrong", "quantiles-of-runs"),
    ("seed-summary", _s(draw="rng.permutation(losses)"), "wrong", "no-replacement"),
    ("seed-summary", _s(cut="np.quantile(means, [0.01, 0.99])"), "wrong", "ninety-eight"),
    ("seed-summary", _s(draw="rng.choice(losses)"), "wrong", "quantiles-of-runs"),
    ("seed-summary", _s(ret="[np.mean(losses), np.quantile(means, [0.025, 0.975])]"), "correct", None),
    ("seed-summary", _HEAD + "    means = []\n    for _ in range(n_boot):\n        rng = np.random.default_rng(seed)\n"
     "        means.append(np.mean(rng.choice(losses, size=len(losses))))\n    low, high = np.quantile(means, [0.025, 0.975])\n"
     "    return [np.mean(losses), low, high]\n", "wrong", "same-resample"),
    ("seed-summary", _START.replace("    means = []\n    ...\n", "    means = []\n    for _ in range(n_boot):\n        means.append(np.mean(rng.choice(losses, size=len(losses))))\n"), "wrong", "dots-left"),
]


def summary_of(text: str | None):
    """The learner's seed_summary from the code of their box, when it is right, else the course's: (function, theirs)."""
    from everything_ai.check import run_code
    if text:
        ran = run_code(text)
        fn = ran.names.get("seed_summary") if ran.error is None else None
        if callable(fn):
            return fn, True
    return run_code(_s()).names["seed_summary"], False
