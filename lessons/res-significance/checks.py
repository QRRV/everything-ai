# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-significance: paired_bootstrap, the mean difference between paired runs and a bootstrap
interval around it. And the course's paired runs: the small GPT of res-hypothesis-control with and without warmup,
seeds 0 to 9, and with 2 and 4 layers at their best learning rate, seeds 0 to 2."""
import itertools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, _show_rounded

# The course's runs of the small GPT of res-hypothesis-control on a CPU with 1 thread: the validation loss after 6
# epochs, for seeds 0 to 9 in that order. Seeds 0 to 4 are those of res-seeds-variance, run again with the same
# numbers; seeds 5 to 9 were run for this lesson.
WARMUP = [2.3902, 2.3924, 2.4174, 2.3938, 2.4012, 2.4034, 2.3864, 2.3989, 2.4313, 2.4005]
NO_WARMUP = [2.3866, 2.3841, 2.4115, 2.3916, 2.3962, 2.3899, 2.3816, 2.3936, 2.4242, 2.3969]
# The 2-layer and the 4-layer GPT of res-fair-baselines, each at its best learning rate, 0.03, seeds 0 to 2.
TWO_LAYERS = [1.7859, 1.7773, 1.7738]
FOUR_LAYERS = [1.7745, 1.7776, 1.7776]


def differences(a, b) -> np.ndarray:
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


def flip_means(diffs) -> np.ndarray:
    """The mean of the differences for every pattern of signs: all that chance alone can give when the change makes no
    difference. 2**n patterns, so only for a handful of pairs."""
    d = np.asarray(diffs, dtype=float)
    return np.array([np.mean(np.array(s) * d) for s in itertools.product([-1, 1], repeat=len(d))])


def flip_p(diffs) -> float:
    """The exact two-sided p-value of the sign-flip test: the share of sign patterns whose mean lies at least as far
    from 0 as the measured one."""
    means = flip_means(diffs)
    return float(np.mean(np.abs(means) >= abs(np.mean(diffs)) - 1e-12))


def bootstrap(a, b, paired: bool = True, n_boot: int = 10000, seed: int = 0) -> tuple:
    """The course's own paired bootstrap, for the demo, and the unpaired one that resamples each group on its own:
    (mean difference, low, high)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    if paired:
        d = a - b
        means = [rng.choice(d, size=len(d)).mean() for _ in range(n_boot)]
    else:
        means = [rng.choice(a, size=len(a)).mean() - rng.choice(b, size=len(b)).mean() for _ in range(n_boot)]
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(a - b)), float(low), float(high)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def signed(x: float, digits: int, lang: str) -> str:
    """A number with its sign, and a real minus: −0.005, +0.006."""
    return ("−" if x < 0 else "+") + number(abs(x), digits, lang)


def pairs_picture(labels: dict) -> str:
    """Two figures. Left: each seed as a line from its run with warmup to its run without: losses, so purple. Right:
    the means that every pattern of signs of the ten differences gives, as a histogram in grey, with the measured mean
    as a line in ink: a mean difference is a number about the losses, not a loss, as the mean and the interval in
    res-seeds-variance."""
    from everything_ai import chart
    lo, hi = min(WARMUP + NO_WARMUP), max(WARMUP + NO_WARMUP)
    left = chart.Panel((-0.35, 1.35), (lo - 0.008, hi + 0.004), title=labels["left"], xlabel="", ylabel=labels["y"],
                       xticks=[])
    for a, b in zip(WARMUP, NO_WARMUP):
        left.line([0, 1], [a, b], "loss", 1.4)
        left.points([0, 1], [a, b], "loss", r=2.6)
    # the names of the two ends, below the lowest run and clear of the axis
    left.text(0, lo - 0.0052, labels["with"], "ink-2")
    left.text(1, lo - 0.0052, labels["without"], "ink-2")
    means = flip_means(differences(WARMUP, NO_WARMUP))
    edges = np.linspace(-0.0075, 0.0075, 31)
    counts, _ = np.histogram(means, bins=edges)
    observed = float(np.mean(differences(WARMUP, NO_WARMUP)))
    right = chart.Panel((edges[0], edges[-1]), (0, counts.max() * 1.15), title=labels["right"], xlabel=labels["x"], ylabel=labels["count"],
                        xticks=[-0.006, -0.003, 0, 0.003, 0.006])
    for c, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        if c:
            right.rect(x0, 0, x1, c, "ink-3")
    right.line([observed, observed], [0, counts.max() * 1.05], "ink", 2.2)
    right.text(observed, counts.max() * 1.1, labels["yours"], "ink", "end")
    # two figures side by side that wrap below each other in a narrow window, each with its own half of the alt text
    figs = [chart.figure([panel], alt=alt, width=330, height=270) for panel, alt in ((left, labels["alt_left"]), (right, labels["alt_right"]))]
    if labels.get("lang") == "nl":
        figs = [chart.with_decimal_comma(f) for f in figs]
    return '<div style="display:flex;flex-wrap:wrap;gap:0 1.5rem">' + "".join(f'<div style="flex:1 1 280px;max-width:330px">{f}</div>' for f in figs) + "</div>"


class _Runs(list):
    """The runs of one setting, which a check message calls by their name, `with_warmup`, instead of writing out all
    ten losses. The learner's code gets a plain list."""

    def __init__(self, name: str, data):
        super().__init__(data)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


# --- exercise 1: paired_bootstrap ---------------------------------------------------------------------------------------------

_HEAD = "import numpy as np\n\n\ndef paired_bootstrap(a, b, n_boot, seed):\n"
_START = (_HEAD + "    # the difference of each pair: the run in a minus the run in b with the same seed\n    diffs = ...\n"
          "    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n"
          "        # the mean of a resample of the differences, drawn with replacement\n        ...\n"
          "    low, high = np.quantile(means, [0.025, 0.975])\n    return [np.mean(diffs), low, high]\n")


def _s(diffs: str = "np.array(a) - np.array(b)", draw: str = "means.append(np.mean(rng.choice(diffs, size=len(diffs))))",
       ret: str = "[np.mean(diffs), low, high]", cut: str = "np.quantile(means, [0.025, 0.975])") -> str:
    return (_HEAD + f"    diffs = {diffs}\n    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n"
            f"        {draw}\n    low, high = {cut}\n    return {ret}\n")


_UNPAIRED = "means.append(np.mean(rng.choice(a, size=len(a))) - np.mean(rng.choice(b, size=len(b))))"


def _three(got) -> list | None:
    """Three numbers, also when they come as the mean and a pair: [mean, [low, high]] is the same answer."""
    try:
        parts = got if isinstance(got, (list, tuple)) else [got]
        x = np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts]) if parts else np.zeros(0)
    except (TypeError, ValueError):
        return None
    return x.tolist() if x.shape == (3,) and np.all(np.isfinite(x)) else None


def _like(got, want) -> bool:
    """The same mean difference, and ends of the interval within what another way of drawing gives: a bootstrap is
    random."""
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

# The lesson is done when these are correct. "three-seeds" is the choice question, "report" the open question.
REQUIRED = ["paired-bootstrap", "three-seeds", "report"]

EXERCISES = {
    "paired-bootstrap": FunctionBox(
        name="paired-bootstrap",
        function="paired_bootstrap",
        start=_START,
        solution=_s(),
        visible=[(_Runs("with_warmup", WARMUP), _Runs("without_warmup", NO_WARMUP), 10000, 0)],
        # The six differences of the last case all differ. With only three distinct values, as before, the ends of the
        # interval jumped between them, and a correct answer that draws another way missed the 8% in about 1 of 10 tries.
        hidden=[(_Runs("four_layers", FOUR_LAYERS), _Runs("two_layers", TWO_LAYERS), 10000, 1), ([2.31, 2.28, 2.35, 2.30], [2.36, 2.33, 2.41, 2.34], 10000, 2),
                ([1.52, 1.61, 1.47, 1.58, 1.55, 1.50], [1.508, 1.589, 1.451, 1.567, 1.507, 1.496], 10000, 3)],
        show=_show,
        compare=_like,
        first=[_DOTS],
        mistakes=[
            Mistake("list-minus", error=lambda e: isinstance(e, TypeError) and "for -: 'list' and 'list'" in str(e),
                    en="`{call}` stops: `a` and `b` are lists, and Python cannot subtract two lists with `-`; that only works for NumPy arrays. Make them arrays first: `np.array(a) - np.array(b)`.",
                    nl="`{call}` stopt: `a` en `b` zijn lijsten, en Python kan twee lijsten niet van elkaar aftrekken met `-`; dat werkt alleen bij NumPy-arrays. Maak er eerst arrays van: `np.array(a) - np.array(b)`."),
            Mistake("means-empty", error=lambda e: isinstance(e, IndexError) and "size 0" in str(e),
                    en="`{call}` stops at `np.quantile`: `means` is still empty. The line in your loop works out the mean of a resample, but does not add it to `means`. Add it with `means.append(...)`.",
                    nl="`{call}` stopt bij `np.quantile`: `means` is nog leeg. De regel in je loop rekent het gemiddelde van een resample uit, maar voegt het niet toe aan `means`. Voeg het toe met `means.append(...)`."),
            Mistake("mean-of-means", wrong=_s(ret="[np.mean(means), low, high]"),
                    en="With `{call}`, your interval is right, but your first number is not quite: it is the mean of the resampled means, which moves a little with the draws. The first number is the mean of the differences themselves: `np.mean(diffs)`.",
                    nl="Met `{call}` klopt je interval, maar je eerste getal net niet: dat is het gemiddelde van de gemiddelden van de resamples, en dat verschuift een beetje met de trekkingen. Het eerste getal is het gemiddelde van de verschillen zelf: `np.mean(diffs)`."),
            Mistake("unpaired", wrong=_s(draw=_UNPAIRED),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your interval is much wider, because you resample `a` and `b` each on its own and so break up the pairs. Then the differences between seeds count as noise. Resample the differences of the pairs: `rng.choice(diffs, size=len(diffs))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je interval is veel breder, omdat je `a` en `b` elk apart resamplet en zo de paren uit elkaar haalt. Dan tellen de verschillen tussen seeds als ruis. Resample de verschillen van de paren: `rng.choice(diffs, size=len(diffs))`."),
            Mistake("order", wrong=_s(diffs="np.array(b) - np.array(a)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your differences have the wrong sign. Take each run in `a` minus the run in `b` with the same seed: `np.array(a) - np.array(b)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je verschillen hebben het verkeerde teken. Neem elke run in `a` min de run in `b` met dezelfde seed: `np.array(a) - np.array(b)`."),
            Mistake("no-replacement", wrong=_s(draw="means.append(np.mean(rng.permutation(diffs)))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every resample holds each difference exactly once, so every mean is the same. Draw with replacement: `rng.choice(diffs, size=len(diffs))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke resample bevat elk verschil precies één keer, dus elk gemiddelde is hetzelfde. Trek met teruglegging: `rng.choice(diffs, size=len(diffs))`."),
            Mistake("one-draw", wrong=_s(draw="means.append(np.mean(rng.choice(diffs)))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each resample is one difference, so your interval runs about from the smallest difference to the largest. A resample holds as many differences as there are pairs: `size=len(diffs)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke resample is één verschil, dus je interval loopt ongeveer van het kleinste verschil tot het grootste. Een resample bevat evenveel verschillen als er paren zijn: `size=len(diffs)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the first with the differences `a` minus `b`, pair by pair; the second with the line that adds the mean of a resample of those differences to `means`.",
                      "nl": "Vervang de twee `...`: de eerste door de verschillen `a` min `b`, paar voor paar; de tweede door de regel die het gemiddelde van een resample van die verschillen aan `means` toevoegt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the two lists of runs, the number of resamples and a seed, and the `def` line of your function has another number of names between its parentheses. Keep `(a, b, n_boot, seed)`.",
                          "nl": "De check roept `{call}` aan, met de twee lijsten runs, het aantal resamples en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b, n_boot, seed)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the differences `a` minus `b` pair by pair, the means of `n_boot` resamples of those differences, drawn with replacement, and the quantiles of those means at 0.025 and 0.975?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je de verschillen `a` min `b` paar voor paar, de gemiddelden van `n_boot` resamples van die verschillen, met teruglegging getrokken, en de kwantielen van die gemiddelden bij 0,025 en 0,975?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of pairs, also when `a` is the lower one and the differences are negative?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het voor elk aantal paren, ook als `a` de lagere is en de verschillen negatief zijn?"},
            "correct": {"en": "Your function is right in all {n} cases: the mean difference of the pairs, and the interval in which it would mostly land with other seeds.",
                        "nl": "Je functie klopt in alle {n} gevallen: het gemiddelde verschil van de paren, en het interval waarin het met andere seeds meestal zou landen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("paired-bootstrap", _START, "empty", None),
    ("paired-bootstrap", _s(), "correct", None),
    ("paired-bootstrap", _s(diffs="[x - y for x, y in zip(a, b)]"), "correct", None),
    ("paired-bootstrap", _s(draw="i = rng.integers(0, len(diffs), size=len(diffs))\n        means.append(np.mean(diffs[i]))"), "correct", None),
    ("paired-bootstrap", _s(ret="(np.mean(diffs), low, high)"), "correct", None),
    ("paired-bootstrap", _HEAD + "    diffs = np.array(a) - np.array(b)\n    rng = np.random.default_rng(seed)\n"
     "    means = rng.choice(diffs, size=(n_boot, len(diffs))).mean(axis=1)\n    return np.mean(diffs), np.percentile(means, [2.5, 97.5])\n", "correct", None),
    ("paired-bootstrap", _s(diffs="a - b"), "error", "list-minus"),
    ("paired-bootstrap", _s(draw="np.mean(rng.choice(diffs, size=len(diffs)))"), "error", "means-empty"),
    ("paired-bootstrap", _s(ret="[np.mean(means), low, high]"), "wrong", "mean-of-means"),
    ("paired-bootstrap", _s(draw=_UNPAIRED), "wrong", "unpaired"),
    ("paired-bootstrap", _s(diffs="np.array(b) - np.array(a)"), "wrong", "order"),
    ("paired-bootstrap", _s(draw="means.append(np.mean(rng.permutation(diffs)))"), "wrong", "no-replacement"),
    ("paired-bootstrap", _s(draw="means.append(np.mean(rng.choice(diffs)))"), "wrong", "one-draw"),
    ("paired-bootstrap", _START.replace("    diffs = ...\n", "    diffs = np.array(a) - np.array(b)\n"), "wrong", "dots-left"),
]


def paired_bootstrap_of(text: str | None):
    """The learner's paired_bootstrap from the code of their box, when it is right, else the course's: (function, theirs)."""
    from everything_ai.check import run_code
    if text:
        ran = run_code(text)
        fn = ran.names.get("paired_bootstrap") if ran.error is None else None
        if callable(fn):
            return fn, True
    return run_code(_s()).names["paired_bootstrap"], False
