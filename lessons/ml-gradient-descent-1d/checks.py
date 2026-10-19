# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-gradient-descent-1d: gradient descent for the bias term, as a loop in a box. And what the
building block of the lesson takes: the loss for every w, its slope, and where gradient descent starts."""
import re

from everything_ai.check import CodeBox, Mistake

# The houses that were sold, from the lesson ml-vocabulary: area in m², price in thousands of euros.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
H = 0.001
LR = 0.00005   # the learning rate of the building block and the worked example
START = 2.5


def loss(w: float, b: float = 50) -> float:
    """The mean squared error of the model w · area + b on the houses."""
    return sum((p - (w * a + b)) ** 2 for a, p in zip(AREA, PRICE)) / len(AREA)


def slope(w: float, b: float = 50) -> float:
    """The slope of the loss for w, measured as the lessons do: with a tiny step h = 0.001."""
    return (loss(w + H, b) - loss(w, b)) / H


def path(steps: int) -> list[float]:
    """w after 0, 1, …, steps update steps of gradient descent, from w = 2.5 with the learning rate of the lesson."""
    ws = [START]
    for _ in range(steps):
        ws.append(ws[-1] - LR * slope(ws[-1]))
    return ws


# The lesson is done when these are correct. "one-step" is the choice question; "why-smaller" is the open question,
# which does not hold anyone back.
REQUIRED = ["one-step", "descend-b"]

_DATA = ("import numpy as np\n\narea = np.array([62, 85, 110, 48, 95, 130, 73, 58])\n"
         "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])\n\n\n"
         "def loss(w, b):\n    residuals = price - (w * area + b)\n    return (residuals ** 2).mean()\n\n\n")
_HEAD = _DATA + "w = 3\nb = 0\nlr = 0.2\nh = 0.001\nfor step in range(25):\n"
_START = _HEAD + "    ...\nb\n"
_BODY = "    slope = (loss(w, b + h) - loss(w, b)) / h\n    b = b - lr * slope\n"
_SOLUTION = _HEAD + _BODY + "b"


def _body(new: str) -> str:
    return _HEAD + new + "b"


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _close(got, want) -> bool:
    """A measured slope depends a little on h: a smaller h, or a step to both sides, is right too."""
    try:
        return abs(float(got) - float(want)) <= 0.001 * abs(float(want)) + 0.01
    except (TypeError, ValueError, OverflowError):
        return False


EXERCISES = {
    "descend-b": CodeBox(
        name="descend-b",
        start=_START,
        solution=_SOLUTION,
        keep=["area = np.array([62, 85, 110, 48, 95, 130, 73, 58])", "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])",
              "def loss(w, b):", "    residuals = price - (w * area + b)", "    return (residuals ** 2).mean()",
              "w = 3", "b = 0", "lr = 0.2", "for step in range(25):"],
        # The check also runs the box with a smaller learning rate: then 25 steps do not reach the bottom yet.
        others=[{"lr": 0.05}],
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("uphill", wrong=_body("    slope = (loss(w, b + h) - loss(w, b)) / h\n    b = b + lr * slope\n"),
                    en="Your last line gives `{got}`: b runs away from the bottom. Every step goes with the slope, uphill. An update step goes against it: `b - lr * slope`, with the slope measured as the loss a step further minus the loss where you are.",
                    nl="Je laatste regel geeft `{got}`: b loopt weg van de bodem. Elke stap gaat met de helling mee, bergop. Een update-stap gaat ertegenin: `b - lr * slope`, met als helling de loss een stap verder min de loss waar je staat."),
            Mistake("once", wrong=_body("    slope = (loss(w, b + h) - loss(w, b)) / h\nb = b - lr * slope\n"),
                    en="Your last line gives `{got}`: b took only one step. Is the line `b = ...` inside the loop, indented like the line above it?",
                    nl="Je laatste regel geeft `{got}`: b heeft maar één stap gezet. Staat de regel `b = ...` in de loop, net zo ver ingesprongen als de regel erboven?"),
            Mistake("no-divide", wrong=_body("    slope = loss(w, b + h) - loss(w, b)\n    b = b - lr * slope\n"),
                    en="Your last line gives `{got}`: your slope is only the change in the loss. A slope is that change divided by the step: divide it by `h`.",
                    nl="Je laatste regel geeft `{got}`: je helling is alleen de verandering van de loss. Een helling is die verandering gedeeld door de stap: deel hem door `h`."),
            Mistake("no-lr", wrong=_body("    slope = (loss(w, b + h) - loss(w, b)) / h\n    b = b - slope\n"),
                    en="Your last line gives `{got}`: without the learning rate, every step is the whole slope, and b jumps back and forth across the valley. Multiply the slope by `lr`.",
                    nl="Je laatste regel geeft `{got}`: zonder learning rate is elke stap de hele helling, en springt b heen en weer over het dal. Vermenigvuldig de helling met `lr`."),
            Mistake("slope-w", wrong=_body("    slope = (loss(w + h, b) - loss(w, b)) / h\n    b = b - lr * slope\n"),
                    en="Your last line gives `{got}`: you change b with the slope for w. Measure the slope for b: change b by `h`, and leave w as it is.",
                    nl="Je laatste regel geeft `{got}`: je verandert b met de helling voor w. Meet de helling voor b: verander b met `h`, en laat w zoals hij is."),
            Mistake("no-update", wrong=_body("    slope = (loss(w, b + h) - loss(w, b)) / h\n"),
                    en="Your last line gives `{got}`: b never changes. Each round, measure the slope and then give b its new value: `b = ...`.",
                    nl="Je laatste regel geeft `{got}`: b verandert nooit. Meet elke ronde de helling en geef b dan zijn nieuwe waarde: `b = ...`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one update step: the slope for b, and then the new b.",
                      "nl": "Vervang de `...` door één update-stap: de helling voor b, en dan de nieuwe b."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another learning rate by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere learning rate."},
            "no_value": {"en": "The last line of the box shows nothing. Put `b` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `b` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and after 25 steps b should be close to the bottom of the valley. Does each round measure the slope for b, and move b against it by `lr` times the slope?",
                      "nl": "Je laatste regel geeft `{got}`, en na 25 stappen hoort b dicht bij de bodem van het dal te zijn. Meet je elke ronde de helling voor b, en schuif je b daartegenin met `lr` keer de helling?"},
            "other": {"en": "Right for `lr = 0.2`. But with {where}, your last line gives `{got}`, and it should give about `53.6`. Do you use `lr` itself in each step?",
                      "nl": "Goed voor `lr = 0.2`. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `53.6` zijn. Gebruik je `lr` zelf in elke stap?"},
            "other_error": {"en": "Right for `lr = 0.2`. But with {where}, your code stops with an error. Do you use `lr` itself in each step?",
                            "nl": "Goed voor `lr = 0.2`. Maar met {where} stopt je code met een fout. Gebruik je `lr` zelf in elke stap?"},
            "correct": {"en": "After 25 steps, b is about 57.75: the bottom of the valley. With the slider in *Fitting a line by hand*, in steps of 1, you found 58. The check also tried `lr = 0.05`: then the steps are smaller, and after 25 steps b is only at about 53.6.",
                        "nl": "Na 25 stappen is b ongeveer 57,75: de bodem van het dal. Met de slider in *Een lijn passen met de hand*, in stappen van 1, vond je 58. De check probeerde ook `lr = 0.05`: dan zijn de stappen kleiner, en is b na 25 stappen pas bij ongeveer 53,6."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("descend-b", _START, "empty", None),
    ("descend-b", _SOLUTION, "correct", None),
    ("descend-b", _body("    b = b - lr * (loss(w, b + h) - loss(w, b)) / h\n"), "correct", None),
    ("descend-b", _body("    slope = (loss(w, b + h) - loss(w, b - h)) / (2 * h)\n    b = b - lr * slope\n"), "correct", None),
    ("descend-b", _body("    slope = (loss(w, b + h) - loss(w, b)) / h\n    b = b + lr * slope\n"), "wrong", "uphill"),
    ("descend-b", _body("    slope = (loss(w, b + h) - loss(w, b)) / h\n    b = b - slope\n"), "wrong", "no-lr"),
    ("descend-b", _body("    slope = (loss(w + h, b) - loss(w, b)) / h\n    b = b - lr * slope\n"), "wrong", "slope-w"),
    ("descend-b", _body("    slope = (loss(w, b + h) - loss(w, b)) / h\n"), "wrong", "no-update"),
    ("descend-b", _body("    b = 57.75\n"), "almost", "other-value"),
    ("descend-b", _body("    slope = (loss(w, b + h) - loss(w, b)) / h\n    b = b - 0.2 * slope\n"), "almost", "other-value"),
    ("descend-b", _HEAD + _BODY + "print(b)", "wrong", "print-last"),
    ("descend-b", _body("    slope = (loss(w, b + h) - loss(w, b)) / h\nb = b - lr * slope\n"), "wrong", "once"),
    ("descend-b", _body("    slope = loss(w, b + h) - loss(w, b)\n    b = b - lr * slope\n"), "wrong", "no-divide"),
    ("descend-b", _body("    slope = (loss(w, b) - loss(w, b + h)) / h\n    b = b - lr * slope\n"), "wrong", "uphill"),
]
