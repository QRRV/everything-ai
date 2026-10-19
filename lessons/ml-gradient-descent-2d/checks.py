# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-gradient-descent-2d: gradient descent for w and b together, as a loop in a box. And the
picture: the loss landscape of the houses as a contour plot, with the path gradient descent takes across it."""
import re

from everything_ai.check import CodeBox, Mistake

# The houses that were sold, from the lesson ml-vocabulary: area in hundreds of m², price in thousands of euros.
AREA = [0.62, 0.85, 1.10, 0.48, 0.95, 1.30, 0.73, 0.58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
H = 0.001
LR = 0.3
LOWEST = (314.5, 45.77)   # where the loss is lowest, worked out with numpy.linalg.lstsq


def loss(w: float, b: float) -> float:
    """The mean squared error of the model w · area + b on the houses, with the area in hundreds of m²."""
    return sum((p - (w * a + b)) ** 2 for a, p in zip(AREA, PRICE)) / len(AREA)


def gradient(w: float, b: float) -> tuple[float, float]:
    return (loss(w + H, b) - loss(w, b)) / H, (loss(w, b + H) - loss(w, b)) / H


def path(steps: int, lr: float = LR) -> list[tuple[float, float]]:
    """Where gradient descent is after 0, 1, …, steps update steps from w = 0, b = 0."""
    out = [(0.0, 0.0)]
    for _ in range(steps):
        w, b = out[-1]
        gw, gb = gradient(w, b)
        out.append((w - lr * gw, b - lr * gb))
    return out


def picture(labels: dict[str, str]) -> str:
    """The contour plot of the loss over w and b, with the path of 200 steps of gradient descent from (0, 0); the
    places after 0, 1, 10, 50 and 200 steps are marked with their number."""
    import matplotlib.pyplot as plt
    import numpy as np
    from everything_ai import plots
    a, p = np.array(AREA), np.array(PRICE)
    W, B = np.meshgrid(np.linspace(-20, 420, 180), np.linspace(-60, 260, 180))
    Z = ((p - (W[..., None] * a + B[..., None])) ** 2).mean(-1)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.contour(W, B, Z, levels=[100, 300, 1000, 3000, 10000, 30000, 60000], colors=[plots.LOSS], linewidths=1, linestyles="-")
    steps = path(200)
    ax.plot([w for w, _ in steps], [b for _, b in steps], color=plots.INK_3, linestyle="-", linewidth=1)
    ax.scatter([w for w, _ in steps], [b for _, b in steps], color=plots.PARAM, s=8, zorder=3)
    for k in (0, 1, 10, 50, 200):
        w, b = steps[k]
        ax.scatter([w], [b], color=plots.PARAM, s=36, zorder=4)
        ax.text(w + 8, b + 6, labels["step"].format(k=k), color=plots.PARAM_TEXT, fontsize=9.5)
    ax.scatter([LOWEST[0]], [LOWEST[1]], color=plots.LOSS, marker="x", s=40, zorder=5)
    ax.set_xlim(-20, 420)
    ax.set_ylim(-60, 260)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title(labels["title"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "one-update" and "valley-floor" are the choice questions in the lesson.
REQUIRED = ["one-update", "descend-wb", "valley-floor"]

_DATA = ("import numpy as np\n\narea = np.array([62, 85, 110, 48, 95, 130, 73, 58]) / 100\n"
         "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])\n\n\n"
         "def loss(w, b):\n    residuals = price - (w * area + b)\n    return (residuals ** 2).mean()\n\n\n")
_HEAD = _DATA + "w = 0\nb = 0\nlr = 0.3\nh = 0.001\nfor step in range(50):\n"
_START = _HEAD + "    ...\n[w, b]\n"
_BODY = ("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
         "    w = w - lr * slope_w\n    b = b - lr * slope_b\n")
_SOLUTION = _HEAD + _BODY + "[w, b]"


def _body(new: str) -> str:
    return _HEAD + new + "[w, b]"


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _close(got, want) -> bool:
    """Where w and b ended, each measured with a tiny step: a smaller h, or a step to both sides, is right too."""
    try:
        got = [float(v) for v in got]
    except (TypeError, ValueError, OverflowError):
        return False
    return len(got) == len(want) == 2 and all(abs(g - v) <= 0.002 * abs(v) + 0.05 for g, v in zip(got, want))


EXERCISES = {
    "descend-wb": CodeBox(
        name="descend-wb",
        start=_START,
        solution=_SOLUTION,
        keep=["area = np.array([62, 85, 110, 48, 95, 130, 73, 58]) / 100", "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])",
              "def loss(w, b):", "    residuals = price - (w * area + b)", "    return (residuals ** 2).mean()",
              "w = 0", "b = 0", "lr = 0.3", "for step in range(50):"],
        # The check also runs the box with a smaller learning rate: a list typed in by hand does not change with it.
        others=[{"lr": 0.1}],
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("one-after-other", wrong=_body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n"
                                                   "    slope_b = (loss(w, b + h) - loss(w, b)) / h\n    b = b - lr * slope_b\n"),
                    en="Your last line gives `{got}`. That goes downhill too, but it is not one step against the gradient: your code measures the slope for b after w has already moved. Measure both slopes first, at the same w and b, and only then update both.",
                    nl="Je laatste regel geeft `{got}`. Dat gaat ook bergaf, maar het is niet één stap tegen de gradient in: je code meet de helling voor b nadat w al verschoven is. Meet eerst beide hellingen, bij dezelfde w en b, en update ze pas daarna allebei."),
            Mistake("one-after-other-b", wrong=_body("    slope_b = (loss(w, b + h) - loss(w, b)) / h\n    b = b - lr * slope_b\n"
                                                     "    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n"),
                    en="Your last line gives `{got}`: your code measures the slope for w after b has already moved. Measure both slopes first, at the same w and b, and only then update both.",
                    nl="Je laatste regel geeft `{got}`: je code meet de helling voor w nadat b al verschoven is. Meet eerst beide hellingen, bij dezelfde w en b, en update ze pas daarna allebei."),
            Mistake("once", wrong=_body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
                                        "w = w - lr * slope_w\nb = b - lr * slope_b\n"),
                    en="Your last line gives `{got}`: w and b took only one step. Are the lines `w = ...` and `b = ...` inside the loop, indented like the lines above them?",
                    nl="Je laatste regel geeft `{got}`: w en b hebben maar één stap gezet. Staan de regels `w = ...` en `b = ...` in de loop, net zo ver ingesprongen als de regels erboven?"),
            Mistake("no-lr", wrong=_body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
                                         "    w = w - slope_w\n    b = b - slope_b\n"),
                    en="Your last line gives `{got}`: without the learning rate, every step is the whole slope, and w and b jump across the valley, further each time. Multiply each slope by `lr`.",
                    nl="Je laatste regel geeft `{got}`: zonder learning rate is elke stap de hele helling, en springen w en b over het dal heen, steeds verder. Vermenigvuldig elke helling met `lr`."),
            Mistake("no-divide", wrong=_body("    slope_w = loss(w + h, b) - loss(w, b)\n    slope_b = loss(w, b + h) - loss(w, b)\n"
                                             "    w = w - lr * slope_w\n    b = b - lr * slope_b\n"),
                    en="Your last line gives `{got}`: your slopes are only the change in the loss. A slope is that change divided by the step: divide each by `h`.",
                    nl="Je laatste regel geeft `{got}`: je hellingen zijn alleen de verandering van de loss. Een helling is die verandering gedeeld door de stap: deel elk door `h`."),
            Mistake("uphill", wrong=_body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
                                          "    w = w + lr * slope_w\n    b = b + lr * slope_b\n"),
                    en="Your last line gives `{got}`: w and b run away from the lowest point, uphill. An update step goes against the gradient: minus.",
                    nl="Je laatste regel geeft `{got}`: w en b lopen weg van het laagste punt, bergop. Een update-stap gaat tegen de gradient in: min."),
            Mistake("only-w", wrong=_body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n"),
                    en="Your last line gives `{got}`: only w moved, and b stayed 0. Update b too, with its own slope.",
                    nl="Je laatste regel geeft `{got}`: alleen w is verschoven, en b bleef 0. Update b ook, met zijn eigen helling."),
            Mistake("one-slope", wrong=_body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n"
                                             "    w = w - lr * slope_w\n    b = b - lr * slope_w\n"),
                    en="Your last line gives `{got}`: b moves with the slope for w. Each parameter needs its own slope: change b by `h` for the slope for b.",
                    nl="Je laatste regel geeft `{got}`: b schuift op met de helling voor w. Elke parameter heeft zijn eigen helling nodig: verander b met `h` voor de helling voor b."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one update step for both parameters: the two slopes, and then the new w and b.",
                      "nl": "Vervang de `...` door één update-stap voor beide parameters: de twee hellingen, en dan de nieuwe w en b."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another learning rate by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere learning rate."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[w, b]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[w, b]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does each round measure both slopes at the same w and b, and move each parameter against its own slope, by `lr` times that slope?",
                      "nl": "Je laatste regel geeft `{got}`. Meet je elke ronde beide hellingen bij dezelfde w en b, en schuif je elke parameter tegen zijn eigen helling in, met `lr` keer die helling?"},
            "other": {"en": "Right for `lr = 0.3`. But with {where}, your last line gives `{got}`, and it should give about `[208.7, 136.8]`. Do you use `lr` itself in each step?",
                      "nl": "Goed voor `lr = 0.3`. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `[208.7, 136.8]` zijn. Gebruik je `lr` zelf in elke stap?"},
            "other_error": {"en": "Right for `lr = 0.3`. But with {where}, your code stops with an error. Do you use `lr` itself in each step?",
                            "nl": "Goed voor `lr = 0.3`. Maar met {where} stopt je code met een fout. Gebruik je `lr` zelf in elke stap?"},
            "correct": {"en": "After 50 steps, w is about 267.6 and b about 86.1, with a loss of about 251. Both moved together, and they are on their way to the lowest point at about 314.5 and 45.8. The check also tried `lr = 0.1`: with smaller steps, they do not get as far.",
                        "nl": "Na 50 stappen is w ongeveer 267,6 en b ongeveer 86,1, met een loss van ongeveer 251. Ze schoven samen op, en zijn op weg naar het laagste punt bij ongeveer 314,5 en 45,8. De check probeerde ook `lr = 0.1`: met kleinere stappen komen ze minder ver."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("descend-wb", _START, "empty", None),
    ("descend-wb", _SOLUTION, "correct", None),
    ("descend-wb", _body("    now = loss(w, b)\n    slope_w = (loss(w + h, b) - now) / h\n    slope_b = (loss(w, b + h) - now) / h\n"
                         "    w, b = w - lr * slope_w, b - lr * slope_b\n"), "correct", None),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w - h, b)) / (2 * h)\n    slope_b = (loss(w, b + h) - loss(w, b - h)) / (2 * h)\n"
                         "    w = w - lr * slope_w\n    b = b - lr * slope_b\n"), "correct", None),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n"
                         "    slope_b = (loss(w, b + h) - loss(w, b)) / h\n    b = b - lr * slope_b\n"), "wrong", "one-after-other"),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
                         "    w = w + lr * slope_w\n    b = b + lr * slope_b\n"), "wrong", "uphill"),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n"), "wrong", "only-w"),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n    b = b - lr * slope_w\n"), "wrong", "one-slope"),
    ("descend-wb", _body("    slope_b = (loss(w, b + h) - loss(w, b)) / h\n    b = b - lr * slope_b\n"
                         "    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    w = w - lr * slope_w\n"), "wrong", "one-after-other-b"),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
                         "w = w - lr * slope_w\nb = b - lr * slope_b\n"), "wrong", "once"),
    ("descend-wb", _body("    slope_w = (loss(w + h, b) - loss(w, b)) / h\n    slope_b = (loss(w, b + h) - loss(w, b)) / h\n"
                         "    w = w - slope_w\n    b = b - slope_b\n"), "wrong", "no-lr"),
    ("descend-wb", _body("    slope_w = loss(w + h, b) - loss(w, b)\n    slope_b = loss(w, b + h) - loss(w, b)\n"
                         "    w = w - lr * slope_w\n    b = b - lr * slope_b\n"), "wrong", "no-divide"),
    ("descend-wb", _body("    w = 267.6\n    b = 86.1\n"), "almost", "other-value"),
    ("descend-wb", _HEAD + _BODY + "print([w, b])", "wrong", "print-last"),
]
