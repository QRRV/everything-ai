# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-learning-rate: a learning rate found with a slider, and a function that says whether
training has converged. And the pictures: three loss curves side by side, and the path and loss curve of one run."""
import re

from everything_ai.check import FunctionBox, Mistake, Value

# The houses that were sold, from the lesson ml-vocabulary: area in m², price in thousands of euros.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
H = 0.001
START = 2.5
STEPS = 10


def loss(w: float, b: float = 50) -> float:
    """The mean squared error of the model w · area + b on the houses."""
    return sum((p - (w * a + b)) ** 2 for a, p in zip(AREA, PRICE)) / len(AREA)


def run(lr: float, steps: int = STEPS) -> tuple[list[float], list[float]]:
    """Gradient descent for w, with b = 50, from w = 2.5: w and the loss before the first step and after each step."""
    ws, losses = [START], [loss(START)]
    for _ in range(steps):
        w = ws[-1]
        ws.append(w - lr * (loss(w + H) - loss(w)) / H)
        losses.append(loss(ws[-1]))
    return ws, losses


def three_curves(labels: dict[str, str]) -> str:
    """Three loss curves side by side: a learning rate that is too small, a good one, and one that is too big."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), gridspec_kw={"wspace": 0.45})
    for ax, lr, title in zip(axes, (0.000005, 0.00005, 0.00015), labels["titles"]):
        _, losses = run(lr)
        ax.plot(range(STEPS + 1), losses, color=plots.LOSS, linestyle="-", marker="o", markersize=5)
        ax.set_ylim(0, max(losses) * 1.08)
        ax.set_xlabel(labels["x"])
        ax.set_title(title)
    axes[0].set_ylabel(labels["y"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def picture(lr: float, labels: dict[str, str]) -> str:
    """Two charts side by side, or one below the other in a narrow window. The loss for every w, with the path of 10
    steps; and the loss curve of the run, where a loss above the chart runs off its top. Drawn as SVG text, for every
    stop of the slider once (L.explore), about a millisecond each."""
    from everything_ai import chart
    ws, losses = run(lr)
    xs = [1.6 + 3.0 * i / 200 for i in range(201)]
    left = chart.Panel((1.6, 4.6), (-1000, 20000), title=labels["left"], xlabel="w", ylabel=labels["loss"])
    left.line(xs, [loss(x) for x in xs], "loss")
    left.line(ws, losses, "ink-3", 1)
    left.points(ws, losses, "param", 3.5)
    right = chart.Panel((0, STEPS), (0, 3000), title=labels["right"], xlabel=labels["x"], ylabel=labels["loss"])
    right.line(range(STEPS + 1), losses, "loss")
    right.points(range(STEPS + 1), losses, "loss", 3)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)   # two fit side by side in a play block


# The lesson is done when these are correct. "too-small" is the choice question in the lesson.
REQUIRED = ["too-small", "pick-lr", "converged"]

_BIGGEST = 0.000105   # the biggest learning rate on the slider that gets the loss below 100 in 10 steps


def _after(lr) -> float:
    return round(run(float(lr))[1][-1], 2)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CONV_START = "def converged(losses):\n    ...\n"
_CONV = "def converged(losses):\n    return abs(losses[-1] - losses[-2]) < 0.01"


def _conv(new: str) -> str:
    return _CONV.replace("    return abs(losses[-1] - losses[-2]) < 0.01", new)


EXERCISES = {
    "pick-lr": Value(
        name="pick-lr",
        start=0.000005,
        solution=_BIGGEST,
        outcome=_after,
        compare=lambda got, want: abs(got - want) < 0.005,
        prose_numbers=True,
        mistakes=[
            Mistake("bigger", state="almost", matches=lambda got, lr: got < 100 and lr < _BIGGEST,
                    en="With this learning rate, the loss after 10 steps is {got}: below 100, so it works. Can the learning rate be bigger, and still get below 100?",
                    nl="Met deze learning rate is de loss na 10 stappen {got}: onder de 100, dus hij werkt. Kan de learning rate groter, en toch onder de 100 komen?"),
            Mistake("too-small", matches=lambda got, lr: got >= 100 and lr < _BIGGEST,
                    en="With this learning rate, the loss after 10 steps is {got}. The loss goes down, but not fast enough: the steps stay on one side of the bottom and are too small. Try a bigger learning rate.",
                    nl="Met deze learning rate is de loss na 10 stappen {got}. De loss daalt, maar niet snel genoeg: de stappen blijven aan één kant van de bodem en zijn te klein. Probeer een grotere learning rate."),
            Mistake("diverges", matches=lambda got, lr: got > 2790,
                    en="With this learning rate, the loss after 10 steps is {got}: higher than at the start. Every jump lands higher than the one before: training diverges. Try a much smaller learning rate.",
                    nl="Met deze learning rate is de loss na 10 stappen {got}: hoger dan bij het begin. Elke sprong komt hoger uit dan de vorige: de training divergeert. Probeer een veel kleinere learning rate."),
            Mistake("too-big", matches=lambda got, lr: got >= 100 and lr > _BIGGEST,
                    en="With this learning rate, the loss after 10 steps is {got}. The steps jump from one side of the bottom to the other, and after 10 steps they have not settled yet. Try a smaller learning rate.",
                    nl="Met deze learning rate is de loss na 10 stappen {got}. De stappen springen van de ene kant van de bodem naar de andere, en zijn na 10 stappen nog niet tot rust gekomen. Probeer een kleinere learning rate."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at the smallest learning rate, where the loss after 10 steps is 665.41. Move it, and watch both charts.",
                      "nl": "De slider staat nog op de kleinste learning rate, waar de loss na 10 stappen 665,41 is. Verschuif hem, en kijk naar beide grafieken."},
            "wrong": {"en": "With this learning rate, the loss after 10 steps is {got}, not below 100.",
                      "nl": "Met deze learning rate is de loss na 10 stappen {got}, niet onder de 100."},
            "correct": {"en": "With this learning rate, the loss after 10 steps is {got}: just below 100, and no bigger learning rate on the slider gets there. One notch more, and the steps still jump at step 10 (100.51); from 0.000135 the loss even grows: training diverges.",
                        "nl": "Met deze learning rate is de loss na 10 stappen {got}: net onder de 100, en geen grotere learning rate op de slider komt daar. Eén stapje verder, en de stappen springen bij stap 10 nog (100,51); vanaf 0,000135 groeit de loss zelfs: de training divergeert."},
        },
    ),
    "converged": FunctionBox(
        name="converged",
        function="converged",
        start=_CONV_START,
        solution=_CONV,
        visible=[([2790.2, 267.4, 110.5, 100.6, 99.98, 99.93],)],
        hidden=[([0.004, 0.003],), ([100.0, 99.995],), ([5.0, 4.995, 3.0],), ([1.0, 3.0, 1.005],), ([5.0, 3.0],), ([10.0, 30.0],), ([1.0, 1.0],), ([50.0, 49.999, 49.9985],), ([7.0, 7.02],), ([2.0, 1.995, 1.0, 0.9995],)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-abs", wrong=_conv("    return losses[-1] - losses[-2] < 0.01"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. The loss dropped by 0.05 in the last step, and a drop is a change too. Look at the size of the change, without its sign: `abs(...)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De loss daalde in de laatste stap met 0,05, en een daling is ook een verandering. Kijk naar de grootte van de verandering, zonder teken: `abs(...)`."),
            Mistake("first-last", wrong=_conv("    return abs(losses[-1] - losses[0]) < 0.01"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function compares the last loss with the first one. Converged means the last step hardly changed the loss: compare the last two.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie vergelijkt de laatste loss met de eerste. Geconvergeerd betekent dat de laatste stap de loss nauwelijks veranderde: vergelijk de laatste twee."),
            Mistake("first-step", wrong=_conv("    return abs(losses[1] - losses[0]) < 0.01"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function looks at the first step. Converged is about the last step: compare the last two losses.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie kijkt naar de eerste stap. Geconvergeerd gaat over de laatste stap: vergelijk de laatste twee losses."),
            Mistake("low-loss", wrong=_conv("    return losses[-1] < 0.01"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function asks whether the loss is small. Converged is about whether it still changes, not about how low it is.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie vraagt of de loss klein is. Geconvergeerd gaat over of hij nog verandert, niet over hoe laag hij is."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with `True` or `False`, worked out from the last two losses, given back with `return`.",
                      "nl": "Vervang de `...` door `True` of `False`, uitgerekend uit de laatste twee losses, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list, and the `def` line of your function has a different number of names between its parentheses. Keep `(losses)`.",
                          "nl": "De check roept `{call}` aan, met één lijst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(losses)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you look at how much the last step changed the loss, up or down, and compare that with 0.01?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Kijk je hoeveel de laatste stap de loss veranderde, omhoog of omlaag, en vergelijk je dat met 0,01?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for lists of any length, when the loss goes up, and when the change is tiny?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor lijsten van elke lengte, als de loss stijgt, en als de verandering piepklein is?"},
            "correct": {"en": "Your function is right in all {n} cases. The run of the example has not converged yet: the last step still changed the loss by 0.05.",
                        "nl": "Je functie klopt in alle {n} gevallen. De run uit het voorbeeld is nog niet geconvergeerd: de laatste stap veranderde de loss nog met 0,05."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner set or typed, state, known mistake).
_C = "def converged(losses):\n"
CASES = [
    ("pick-lr", 0.000005, "empty", None),
    ("pick-lr", 0.000105, "correct", None),
    ("pick-lr", 0.00005, "almost", "bigger"),
    ("pick-lr", 0.0001, "almost", "bigger"),
    ("pick-lr", 0.00001, "wrong", "too-small"),
    ("pick-lr", 0.000025, "wrong", "too-small"),
    ("pick-lr", 0.00011, "wrong", "too-big"),
    ("pick-lr", 0.00013, "wrong", "too-big"),
    ("pick-lr", 0.000135, "wrong", "diverges"),
    ("pick-lr", 0.00015, "wrong", "diverges"),
    ("converged", _CONV_START, "empty", None),
    ("converged", _CONV, "correct", None),
    ("converged", _C + "    change = losses[-1] - losses[-2]\n    return -0.01 < change < 0.01", "correct", None),
    ("converged", _C + "    if abs(losses[len(losses) - 1] - losses[len(losses) - 2]) < 0.01:\n        return True\n    return False", "correct", None),
    ("converged", _conv("    return losses[-1] - losses[-2] < 0.01"), "wrong", "no-abs"),
    ("converged", _conv("    return abs(losses[-1] - losses[0]) < 0.01"), "wrong", "first-last"),
    ("converged", _conv("    return losses[-1] < 0.01"), "wrong", "low-loss"),
    ("converged", _C + "    return False", "almost", None),
    ("converged", _conv("    return abs(losses[1] - losses[0]) < 0.01"), "wrong", "first-step"),
    ("converged", _C + "    for i in range(1, len(losses)):\n        if abs(losses[i] - losses[i - 1]) >= 0.01:\n            return False\n    return True", "almost", None),
]
