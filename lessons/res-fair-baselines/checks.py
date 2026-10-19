# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-fair-baselines: best_setting, the learning rate with the lowest mean validation loss over
the seeds, and that mean. And the course's runs of the small GPT of res-hypothesis-control with 2 and with 4 layers,
each over the same learning rates and seeds."""
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, _show_rounded

# The course's runs of the small GPT of res-hypothesis-control (6 epochs on the first quarter of Tiny Shakespeare,
# warmup 100), on a CPU with 1 thread: the validation loss after the last epoch, by number of layers, by learning rate,
# for seeds 0, 1 and 2. Filled in from the reference runs.
GRID = {
    2: {
        0.001: [2.3902, 2.3924, 2.4174],
        0.002: [2.1933, 2.2405, 2.2115],
        0.003: [2.0827, 2.0987, 2.085],
        0.005: [1.9753, 1.962, 1.9484],
        0.01: [1.8346, 1.8399, 1.8325],
        0.02: [1.7926, 1.7943, 1.7854],
        0.03: [1.7859, 1.7773, 1.7738],
        0.05: [1.7902, 1.826, 1.7874],
        0.1: [1.9491, 2.7611, 1.8622],
    },
    4: {
        0.001: [2.3335, 2.3245, 2.3374],
        0.002: [2.1218, 2.113, 2.1361],
        0.003: [2.0117, 1.9931, 2.0099],
        0.005: [1.908, 1.8888, 1.8944],
        0.01: [1.7963, 1.8029, 1.8168],
        0.02: [1.7858, 1.783, 1.7831],
        0.03: [1.7745, 1.7776, 1.7776],
        0.05: [1.8069, 1.7969, 1.8311],
        0.1: [2.842, 2.7658, 2.7524],
    },
}
DEFAULT_LR = 0.001   # the learning rate of the control run in res-hypothesis-control


def mean_of(layers: int, lr: float) -> float:
    return float(np.mean(GRID[layers][lr]))


def best_of(layers: int) -> tuple:
    """The course's own best_setting, for the picture and the demo: (learning rate, mean)."""
    lr = min(GRID[layers], key=lambda k: np.mean(GRID[layers][k]))
    return lr, mean_of(layers, lr)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def signed(x: float, digits: int, lang: str) -> str:
    """A difference with its sign, and a real minus: −0.623, +0.002."""
    text = number(abs(x), digits, lang)
    return ("\u2212" if x < 0 else "+") + text


def rate(lr: float, lang: str) -> str:
    """A learning rate as the lessons write it: 0.001, in Dutch 0,001."""
    text = f"{lr:g}"
    return text.replace(".", ",") if lang == "nl" else text


def tuning_picture(labels: dict) -> str:
    """The mean validation loss over the seeds against the learning rate, a line for each model: losses, so purple;
    the 2-layer GPT dashed. Hollow points are the single seeds. The best rate of each model has a ring."""
    from everything_ai import chart
    values = [v for g in GRID.values() for vs in g.values() for v in vs]
    rates = sorted(GRID[2])
    top = max(values) + 0.05
    panel = chart.Panel((rates[0] / 1.4, rates[-1] * 1.4), (min(values) - 0.05, top), title=labels["title"],
                        xlabel=labels["x"], ylabel=labels["y"], xlog=True, xticks=[0.001, 0.003, 0.01, 0.03, 0.1])
    for k, (layers, dashed) in enumerate(((2, True), (4, False))):
        means = [mean_of(layers, lr) for lr in rates]
        panel.line(rates, means, "loss", 2.2, dashed=dashed)
        for lr in rates:
            panel.points([lr] * len(GRID[layers][lr]), GRID[layers][lr], "loss", r=2.2, hollow=True)
        best, m = best_of(layers)
        panel.points([best], [m], "loss", r=6, hollow=True)
        y = top - 0.1 - 0.13 * k   # a small legend in the empty corner at the top left
        panel.line([0.0012, 0.0021], [y, y], "loss", 2.2, dashed=dashed)
        panel.text(0.0025, y - 0.03, labels[layers], "ink-2", "start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=270)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


class _Named(dict):
    """The results of one model, which a check message calls by its name, `two_layers`, instead of writing out all
    its rates and losses. The learner's code gets a plain dictionary."""

    def __init__(self, name: str, data: dict):
        super().__init__(data)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {k: list(v) for k, v in self.items()}


# --- exercise 1: best_setting -------------------------------------------------------------------------------------------------

_HEAD = "import numpy as np\n\n\ndef best_setting(results):\n"
_START = (_HEAD + "    best_lr = None\n    best_mean = None\n    for lr in results:\n        # the mean validation loss over the seeds of this learning rate\n"
          "        mean = ...\n        # keep it when it is the first, or lower than the best so far\n        if best_mean is None or ...:\n"
          "            best_lr = lr\n            best_mean = mean\n    return [best_lr, best_mean]\n")


def _b(mean: str = "np.mean(results[lr])", better: str = "mean < best_mean", ret: str = "[best_lr, best_mean]") -> str:
    return (_HEAD + "    best_lr = None\n    best_mean = None\n    for lr in results:\n"
            f"        mean = {mean}\n        if best_mean is None or {better}:\n            best_lr = lr\n            best_mean = mean\n"
            f"    return {ret}\n")


def _pair(got):
    """[rate, mean], also as a tuple, a NumPy array or with NumPy numbers: the same answer."""
    try:
        if isinstance(got, (list, tuple, np.ndarray)) and len(got) == 2:
            return float(got[0]), float(got[1])
    except (TypeError, ValueError):
        pass
    return None


def _same(got, want) -> bool:
    g, w = _pair(got), _pair(want)
    return g is not None and w is not None and abs(g[0] - w[0]) < 1e-12 and abs(g[1] - w[1]) < 1e-6


def _show(v) -> str:
    """The mean with 6 significant digits: a mean rounded to 4 decimals then does not look the same as the right one."""
    p = _pair(v)
    return f"[{p[0]:g}, {p[1]:.6g}]" if p else _show_rounded(v, 4)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# A case where the best single run and the best mean point at different rates, and one where seed 0 does.
_LUCKY = {0.001: [2.05, 2.61, 2.58], 0.003: [2.21, 2.19, 2.24]}
_FIRST_SEED = {0.002: [2.30, 2.12, 2.14], 0.004: [2.25, 2.28, 2.27]}


def _cases():
    two, four = _Named("two_layers", GRID.get(2, {})), _Named("four_layers", GRID.get(4, {}))
    return [(two,)], [(four,), (_LUCKY,), (_FIRST_SEED,), ({0.01: [1.9, 2.0]},), ({0.03: [2.4, 2.5, 2.6], 0.01: [2.0, 2.1, 2.2], 0.003: [2.2, 2.3, 2.4]},)]


# The lesson is done when these are correct. "fair" is the choice question, "review-claim" the open question.
REQUIRED = ["best-setting", "fair", "review-claim"]


def _exercises():
    visible, hidden = _cases()
    return {
        "best-setting": FunctionBox(
            name="best-setting",
            function="best_setting",
            start=_START,
            solution=_b(),
            visible=visible,
            hidden=hidden,
            show=_show,
            compare=_same,
            first=[_DOTS],
            mistakes=[
                Mistake("highest", wrong=_b(better="mean > best_mean"),
                        en="`{call}` gives `{got}`, and it should give `{want}`: you keep the learning rate with the highest loss. A loss is better when it is lower: keep `mean` when it is lower than `best_mean`.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je houdt de learning rate met de hoogste loss. Een loss is beter als hij lager is: houd `mean` als die lager is dan `best_mean`."),
                Mistake("best-run", wrong=_b(mean="min(results[lr])"),
                        en="`{call}` gives `{got}`, and it should give `{want}`: you take the lowest single run of each learning rate, so a lucky seed decides. Take the mean over the seeds: `np.mean(results[lr])`.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt de laagste losse run van elke learning rate, dus een gelukkige seed beslist. Neem het gemiddelde over de seeds: `np.mean(results[lr])`."),
                Mistake("first-seed", wrong=_b(mean="results[lr][0]"),
                        en="`{call}` gives `{got}`, and it should give `{want}`: you look only at the first seed of each learning rate. Take the mean over all its seeds: `np.mean(results[lr])`.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je kijkt alleen naar de eerste seed van elke learning rate. Neem het gemiddelde over al zijn seeds: `np.mean(results[lr])`."),
                Mistake("rate-only", matches=lambda got, text: not isinstance(got, (list, tuple)) and isinstance(got, (int, float)),
                        en="`{call}` gives `{got}`, and it should give `{want}`: give back the learning rate and its mean, as `[best_lr, best_mean]`.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: geef de learning rate en zijn gemiddelde terug, als `[best_lr, best_mean]`."),
                Mistake("swapped", wrong=_b(ret="[best_mean, best_lr]"),
                        en="`{call}` gives `{got}`, and it should give `{want}`: you give back the mean first and the learning rate second. Give back `[best_lr, best_mean]`: the rate first.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je geeft eerst het gemiddelde terug en dan de learning rate. Geef `[best_lr, best_mean]` terug: eerst de rate."),
                Mistake("fixed-count", wrong=_b(mean="sum(results[lr]) / 3"),
                        en="`{call}` gives `{got}`, and it should give `{want}`: you divide by 3, but here a learning rate has 2 seeds. Divide by the number of seeds, `len(results[lr])`, or take `np.mean(results[lr])`.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door 3, maar hier heeft een learning rate 2 seeds. Deel door het aantal seeds, `len(results[lr])`, of neem `np.mean(results[lr])`."),
                # the worked example rounds its means to print them; a mean rounded inside the function is another number
                Mistake("rounded", matches=lambda got, text: re.search(r"\bround\(", text) is not None and _pair(got) is not None,
                        en="`{call}` gives `{got}`, and it should give `{want}`: you round the mean. Give back the mean itself; the worked example rounds only to print it.",
                        nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rondt het gemiddelde af. Geef het gemiddelde zelf terug; het uitgewerkte voorbeeld rondt alleen af om het te printen."),
                # a test without `best_mean is None or` compares the first mean with None
                Mistake("none-compare", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and "not supported between" in str(e),
                        en="With `{call}`, your function stops at the first learning rate: `best_mean` is still `None` there, and Python cannot compare a number with `None`. Keep `best_mean is None or` at the start of the test: then the first rate is always kept.",
                        nl="Met `{call}` stopt je functie bij de eerste learning rate: daar is `best_mean` nog `None`, en Python kan een getal niet met `None` vergelijken. Houd `best_mean is None or` aan het begin van de test: dan wordt de eerste rate altijd bewaard."),
            ],
            texts={
                "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
                "empty": {"en": "Replace the two `...`: the first with the mean of the losses of `lr`; the second with the test that `mean` is lower than `best_mean`.",
                          "nl": "Vervang de twee `...`: de eerste door het gemiddelde van de losses van `lr`; de tweede door de test dat `mean` lager is dan `best_mean`."},
                "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
                "arguments": {"en": "The check calls `{call}`, with the results of one model, and the `def` line of your function has another number of names between its parentheses. Keep `(results)`.",
                              "nl": "De check roept `{call}` aan, met de resultaten van één model, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(results)`."},
                "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the mean over the seeds of each learning rate, and keep the rate with the lowest mean?",
                          "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je het gemiddelde over de seeds van elke learning rate, en houd je de rate met het laagste gemiddelde?"},
                "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any results: any number of seeds per rate, and the rates in any order?",
                           "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor alle resultaten: elk aantal seeds per rate, en de rates in elke volgorde?"},
                "correct": {"en": "Your function is right in all {n} cases: the learning rate with the lowest mean over its seeds, and that mean.",
                            "nl": "Je functie klopt in alle {n} gevallen: de learning rate met het laagste gemiddelde over zijn seeds, en dat gemiddelde."},
            },
        ),
    }


EXERCISES = _exercises()

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("best-setting", _START, "empty", None),
    ("best-setting", _b(), "correct", None),
    ("best-setting", _b(mean="sum(results[lr]) / len(results[lr])"), "correct", None),
    ("best-setting", _b(ret="(best_lr, best_mean)"), "correct", None),
    ("best-setting", _HEAD + "    means = {lr: np.mean(results[lr]) for lr in results}\n    best = min(means, key=means.get)\n    return [best, means[best]]\n", "correct", None),
    ("best-setting", _b(better="mean <= best_mean"), "correct", None),
    ("best-setting", _b(ret="np.array([best_lr, best_mean])"), "correct", None),
    ("best-setting", _HEAD + "    best_lr = None\n    best_mean = float('inf')\n    for lr, losses in results.items():\n        mean = np.mean(losses)\n        if mean < best_mean:\n"
                     "            best_lr = lr\n            best_mean = mean\n    return [best_lr, best_mean]\n", "correct", None),
    ("best-setting", _b(better="mean > best_mean"), "wrong", "highest"),
    ("best-setting", _b(mean="min(results[lr])"), "wrong", "best-run"),
    ("best-setting", _b(mean="results[lr][0]"), "wrong", "first-seed"),
    ("best-setting", _b(ret="best_lr"), "wrong", "rate-only"),
    ("best-setting", _b(ret="[best_mean, best_lr]"), "wrong", "swapped"),
    ("best-setting", _b(mean="sum(results[lr]) / 3"), "wrong", "fixed-count"),
    ("best-setting", _b(mean="round(np.mean(results[lr]), 3)"), "wrong", "rounded"),
    ("best-setting", _b(mean="round(float(np.mean(results[lr])), 4)"), "wrong", "rounded"),
    ("best-setting", _b().replace("best_mean is None or ", ""), "error", "none-compare"),
    ("best-setting", _START.replace("        mean = ...\n", "        mean = np.mean(results[lr])\n"), "wrong", "dots-left"),
]


def best_setting_of(text: str | None):
    """The learner's best_setting from the code of their box, when it is right, else the course's: (function, theirs)."""
    from everything_ai.check import run_code
    if text:
        ran = run_code(text)
        fn = ran.names.get("best_setting") if ran.error is None else None
        if callable(fn):
            return fn, True
    return run_code(_b()).names["best_setting"], False
