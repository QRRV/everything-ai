# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-gradient: the gradient of the loss of the houses as a function. And the contour plot of
that loss over w and b, with the two slopes at one point, or with three points to compare."""
import re

from everything_ai.check import FunctionBox, Mistake

# The houses that were sold, from the lesson ml-vocabulary: area in m², price in thousands of euros.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
H = 0.001
LOWEST = (3.145, 45.77)   # where the loss is lowest, worked out with numpy.linalg.lstsq
POINTS = {"A": (3.0, 50), "B": (2.8, 80), "C": (3.3, 100)}


def loss(w: float, b: float) -> float:
    """The mean squared error of the model w · area + b on the houses."""
    return sum((p - (w * a + b)) ** 2 for a, p in zip(AREA, PRICE)) / len(AREA)


def gradient(w: float, b: float) -> list[float]:
    """The slope for w and the slope for b, each measured with a tiny step h = 0.001."""
    return [(loss(w + H, b) - loss(w, b)) / H, (loss(w, b + H) - loss(w, b)) / H]


def picture(labels: dict[str, str], *, slopes: bool = True) -> str:
    """The contour plot of the loss over w and b: each line joins the (w, b) with the same loss, every 600. With
    `slopes`, the point w = 3, b = 50 with an arrow along w and one along b, each the way the loss goes down; otherwise
    the points A, B and C of the exercise."""
    import matplotlib.pyplot as plt
    import numpy as np
    from everything_ai import plots
    a, p = np.array(AREA), np.array(PRICE)
    W, B = np.meshgrid(np.linspace(2.6, 3.6, 160), np.linspace(-20, 140, 160))
    Z = ((p - (W[..., None] * a + B[..., None])) ** 2).mean(-1)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.contour(W, B, Z, levels=np.arange(150, 9000, 600), colors=[plots.LOSS], linewidths=1, linestyles="-")
    ax.scatter([LOWEST[0]], [LOWEST[1]], color=plots.LOSS, marker="x", s=40, zorder=3)
    ax.text(LOWEST[0] + 0.03, LOWEST[1] - 4, labels["lowest"], color=plots.LOSS_TEXT, fontsize=9.5, va="top")
    if slopes:
        w, b = 3.0, 50
        ax.annotate("", xy=(w + 0.12, b), xytext=(w, b), arrowprops={"arrowstyle": "-|>", "color": plots.GRAD, "lw": 2})
        ax.annotate("", xy=(w, b + 12), xytext=(w, b), arrowprops={"arrowstyle": "-|>", "color": plots.GRAD, "lw": 2})
        ax.text(w + 0.03, b + 5, labels["along_w"], color=plots.GRAD_TEXT, fontsize=9.5, va="bottom", zorder=5,
                bbox={"facecolor": plots.PAGE, "edgecolor": "none", "pad": 1.5})
        ax.text(w - 0.02, b + 14, labels["along_b"], color=plots.GRAD_TEXT, fontsize=9.5, ha="right", zorder=5,
                bbox={"facecolor": plots.PAGE, "edgecolor": "none", "pad": 1.5})
        ax.scatter([w], [b], color=plots.PARAM, s=36, zorder=4)
    else:
        for name, (w, b) in POINTS.items():
            ax.scatter([w], [b], color=plots.PARAM, s=36, zorder=4)
            ax.text(w + 0.02, b + 4, name, color=plots.PARAM_TEXT, fontsize=11, fontweight=600, zorder=5,
                    bbox={"facecolor": plots.PAGE, "edgecolor": "none", "pad": 1.5})
    ax.set_xlim(2.6, 3.6)
    ax.set_ylim(-20, 140)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title(labels["title"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "two-ways" and "steepest" are the choice questions in the lesson.
REQUIRED = ["two-ways", "gradient", "steepest"]

_DATA = ("import numpy as np\n\narea = np.array([62, 85, 110, 48, 95, 130, 73, 58])\n"
         "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])\n\n\n"
         "def loss(w, b):\n    residuals = price - (w * area + b)\n    return (residuals ** 2).mean()\n\n\n")
_START = _DATA + "def gradient(w, b):\n    h = 0.001\n    ...\n"
_SOLUTION = (_DATA + "def gradient(w, b):\n    h = 0.001\n    slope_w = (loss(w + h, b) - loss(w, b)) / h\n"
             "    slope_b = (loss(w, b + h) - loss(w, b)) / h\n    return [slope_w, slope_b]")


def _grad(new: str) -> str:
    return _SOLUTION.replace("    return [slope_w, slope_b]", new)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _close(got, want) -> bool:
    """Two slopes, each measured: a smaller h, a step back, or a step to both sides, is right too. A step of h = 0.001
    measures the slope for w about 7.5 too high, and a step back as much too low, because the loss bends steeply along
    w; along b it bends hardly at all."""
    try:
        got = [float(v) for v in got]
    except (TypeError, ValueError):
        return False
    return len(got) == len(want) == 2 and all(abs(g - v) <= 0.01 * abs(v) + margin
                                              for g, v, margin in zip(got, want, (16, 0.1)))


EXERCISES = {
    "gradient": FunctionBox(
        name="gradient",
        function="gradient",
        start=_START,
        solution=_SOLUTION,
        visible=[(3, 50)],
        hidden=[(3, 70), (2.5, 50), (2.8, 80), (3.3, 100), (3.145, 45.77)],
        compare=_close,
        show=lambda v: [float(f"{float(x):.6g}") for x in v] if isinstance(v, (list, tuple)) and len(v) == 2 else v,
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_grad("    return [slope_b, slope_w]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right two slopes, in the other order. The gradient lists the slopes in the order of the parameters: first w, then b.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de twee goede hellingen, in de andere volgorde. De gradient zet de hellingen in de volgorde van de parameters: eerst w, dan b."),
            Mistake("downhill", wrong=_grad("    return [-slope_w, -slope_b]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every sign is flipped. That is the way downhill; the gradient itself points uphill.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk teken staat andersom. Dat is de richting bergaf; de gradient zelf wijst bergop."),
            Mistake("same-nudge", wrong=_SOLUTION.replace("    slope_b = (loss(w, b + h) - loss(w, b)) / h", "    slope_b = (loss(w + h, b) - loss(w, b)) / h"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: both slopes change w. For the slope for b, change b by `h`, and keep w as it is.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: beide hellingen veranderen w. Voor de helling voor b verander je b met `h`, en laat je w zoals hij is."),
            Mistake("same-nudge-b", wrong=_SOLUTION.replace("    slope_w = (loss(w + h, b) - loss(w, b)) / h", "    slope_w = (loss(w, b + h) - loss(w, b)) / h"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: both slopes change b. For the slope for w, change w by `h`, and keep b as it is.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: beide hellingen veranderen b. Voor de helling voor w verander je w met `h`, en laat je b zoals hij is."),
            Mistake("no-divide", wrong=_SOLUTION.replace("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h",
                                                         "    slope_w = loss(w + h, b) - loss(w, b)\n    slope_b = loss(w, b + h) - loss(w, b)"),
                    en="`{call}` gives `{got}`: the changes in the loss themselves. A slope is that change divided by the step: divide each by `h`.",
                    nl="`{call}` geeft `{got}`: de veranderingen van de loss zelf. Een helling is die verandering gedeeld door de stap: deel ze allebei door `h`."),
            Mistake("one-number", matches=lambda got, text: isinstance(got, (int, float)),
                    en="`{call}` gives one number, `{got}`. The gradient has one slope per parameter: give back a list of two, `[slope_w, slope_b]`.",
                    nl="`{call}` geeft één getal, `{got}`. De gradient heeft één helling per parameter: geef een lijst van twee terug, `[slope_w, slope_b]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope for w and the slope for b, given back as a list with `return`.",
                      "nl": "Vervang de `...` door de helling voor w en de helling voor b, teruggegeven als lijst met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(w, b)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give about `{want}`. Do you measure one slope for each parameter, changing only that one by `h`, and give back both, w first?",
                      "nl": "`{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn. Meet je één helling per parameter, waarbij je alleen die ene met `h` verandert, en geef je ze allebei terug, w eerst?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give about `{want}`. Do you measure at the w and b the function gets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet ongeveer `{want}` zijn. Meet je bij de w en b die de functie krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. At w = 3 and b = 50 the gradient is about [−1472, −15.5]: both slopes are negative, so making w and b a bit bigger lowers the loss. At the lowest point, both slopes are close to 0.",
                        "nl": "Je functie klopt in alle {n} gevallen. Bij w = 3 en b = 50 is de gradient ongeveer `[-1472, -15.5]`: beide hellingen zijn negatief, dus w en b een beetje groter maken verlaagt de loss. Bij het laagste punt liggen beide hellingen dicht bij 0."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_G = _DATA + "def gradient(w, b):\n    h = 0.001\n"
CASES = [
    ("gradient", _START, "empty", None),
    ("gradient", _SOLUTION, "correct", None),
    ("gradient", _G + "    return [(loss(w + h, b) - loss(w - h, b)) / (2 * h), (loss(w, b + h) - loss(w, b - h)) / (2 * h)]", "correct", None),
    ("gradient", _G + "    now = loss(w, b)\n    return [(loss(w + h, b) - now) / h, (loss(w, b + h) - now) / h]", "correct", None),
    ("gradient", _grad("    return [slope_b, slope_w]"), "wrong", "swapped"),
    ("gradient", _G + "    return [(loss(w, b) - loss(w - h, b)) / h, (loss(w, b) - loss(w, b - h)) / h]", "correct", None),
    ("gradient", _SOLUTION.replace("    slope_w = (loss(w + h, b) - loss(w, b)) / h", "    slope_w = (loss(w, b + h) - loss(w, b)) / h"), "wrong", "same-nudge-b"),
    ("gradient", _grad("    return [-slope_w, -slope_b]"), "wrong", "downhill"),
    ("gradient", _SOLUTION.replace("    slope_b = (loss(w, b + h) - loss(w, b)) / h", "    slope_b = (loss(w + h, b) - loss(w, b)) / h"), "wrong", "same-nudge"),
    ("gradient", _SOLUTION.replace("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h",
                                   "    slope_w = loss(w + h, b) - loss(w, b)\n    slope_b = loss(w, b + h) - loss(w, b)"), "wrong", "no-divide"),
    ("gradient", _grad("    return slope_w"), "wrong", "one-number"),
    ("gradient", _G + "    return [-1471.7, -15.5]", "almost", None),
]
