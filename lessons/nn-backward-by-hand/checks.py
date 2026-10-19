# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-backward-by-hand: the backward pass through a neuron with ReLU and a squared loss, by
hand, and checking the slopes of the worked example by moving each parameter. And the picture: the computational
graph of one student's loss, with the backward pass walked one step at a time on a slider."""
import re

from everything_ai import draw
from everything_ai.check import CodeBox, Mistake

# The student of the lesson: x = 4 hours, the grade y = 7, at w = 1 and b = 1.
# Colours: what is learned (w, b), the loss, and the slopes; the rest stays ink, so the picture has three roles.
_NODES = [("w", 0, 0, "w", "1", "param"), ("x", 0, 1, "x", "4", None), ("m", 1, 0.5, "m = w · x", "4", None),
          ("b", 1, 1.5, "b", "1", "param"), ("p", 2, 1, "ŷ = m + b", "5", None), ("r", 3, 1, "r = 7 − ŷ", "2", None),
          ("L", 4, 1, "L = r²", "4", "loss")]
# The step of the backward pass at which each node gets its slope, and the slope.
_SLOPES = {"L": (1, "∂L/∂L = 1"), "r": (2, "∂L/∂r = 4"), "p": (3, "∂L/∂ŷ = −4"), "m": (4, "∂L/∂m = −4"),
           "b": (4, "∂L/∂b = −4"), "w": (5, "∂L/∂w = −16")}
# Each arrow, the step at which the backward pass uses it, and its local gradient.
_EDGES = [("w", "m", 5, "x = 4"), ("x", "m", None, None), ("m", "p", 4, "1"), ("b", "p", 4, "1"), ("p", "r", 3, "−1"),
          ("r", "L", 2, "2r = 4")]


def picture(step: int, labels: dict[str, str]):
    """The computational graph of the student's loss, with the values of the forward pass, and the slopes and local
    gradients the backward pass has reached by `step`, from 0 (none yet) to 5 (all), from top to bottom; in grey, the step
    at which each slope still to come arrives, so the order shows before you touch the slider. Drawn once for every
    step of the slider (L.explore)."""
    nodes = [(key, col, row, label, value, _SLOPES[key][1] if key in _SLOPES and _SLOPES[key][0] <= step else None, role)
             for key, col, row, label, value, role in _NODES]
    edges = [(a, b, local if at is not None and at <= step else None) for a, b, at, local in _EDGES]
    later = {key: labels["later"].format(n=at) for key, (at, _) in _SLOPES.items() if at > step}
    return draw.computational_graph(nodes, edges, title=labels["title"], alt=labels["alt"], down=True,   # five steps deep: top to bottom
                                    later=later)


def known(step: int) -> list[str]:
    """The slopes the backward pass has reached by `step`, written out, for the alt of the picture."""
    return [text for at, text in sorted(_SLOPES.values()) if at <= step]


# The lesson is done when these are correct. "branch" is the choice question in the lesson.
REQUIRED = ["backward", "branch", "check"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _close(got, want) -> bool:
    """Two numbers, each the same as the course's to well within 0.01."""
    try:
        return len(got) == len(want) == 2 and all(abs(float(g) - float(w)) < 0.01 for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


# --- exercise 1: backwards through a neuron with ReLU ---------------------------------------------------------------

_B_HEAD = ("x = 2\ny = 7\nw = 1.5\nb = 1\n\n# the forward pass\nz = w * x + b\nif z > 0:\n    y_pred = z\nelse:\n    y_pred = 0\n"
           "r = y - y_pred\nloss = r ** 2\n\n# the backward pass\n")
_B_END = "[dloss_dw, dloss_db]"
_B_START = (_B_HEAD + "dloss_dr = ...\ndloss_dy_pred = ...\nif z > 0:\n    dloss_dz = ...\nelse:\n    dloss_dz = 0\ndloss_dw = ...\ndloss_db = ...\n"
            + _B_END + "\n")


def _b(dr="2 * r", dy="dloss_dr * -1", dz="dloss_dy_pred * 1", dw="dloss_dz * x", db="dloss_dz * 1") -> str:
    return (_B_HEAD + f"dloss_dr = {dr}\ndloss_dy_pred = {dy}\nif z > 0:\n    dloss_dz = {dz}\nelse:\n    dloss_dz = 0\n"
            f"dloss_dw = {dw}\ndloss_db = {db}\n" + _B_END)


_B = _b()

# --- exercise 3: check it by moving ---------------------------------------------------------------------------------

_C_HEAD = "def loss(w, b):\n    x = 4\n    y = 7\n    return (y - (w * x + b)) ** 2\n\n\nh = 0.0001\n"
_C_END = "[slope_w, slope_b]"
_C_START = _C_HEAD + "slope_w = ...\nslope_b = ...\n" + _C_END + "\n"


def _c(sw="(loss(1 + h, 1) - loss(1, 1)) / h", sb="(loss(1, 1 + h) - loss(1, 1)) / h") -> str:
    return _C_HEAD + f"slope_w = {sw}\nslope_b = {sb}\n" + _C_END


_C = _c()

EXERCISES = {
    "backward": CodeBox(
        name="backward",
        start=_B_START,
        solution=_B,
        compare=_close,
        keep=["z = w * x + b", "r = y - y_pred", "loss = r ** 2", _B_END],
        others=[{"w": -1.0}, {"x": 3, "y": 1, "w": 2, "b": -1}],
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_b(dy="dloss_dr"),
                    en="Your last line gives `{got}`: the right size, but the wrong sign. `r = y - y_pred` goes down when `y_pred` goes up: its local gradient is −1, so `dloss_dy_pred = dloss_dr * -1`.",
                    nl="Je laatste regel geeft `{got}`: de goede grootte, maar het verkeerde teken. `r = y - y_pred` daalt als `y_pred` stijgt: zijn local gradient is −1, dus `dloss_dy_pred = dloss_dr * -1`."),
            Mistake("no-two", wrong=_b(dr="r"),
                    en="Your last line gives `{got}`: half of what it should be. The local gradient of L = r² is 2r, not r.",
                    nl="Je laatste regel geeft `{got}`: de helft van wat het moet zijn. De local gradient van L = r² is 2r, niet r."),
            Mistake("relu-value", wrong=_b(dz="dloss_dy_pred * z"),
                    en="Your last line gives `{got}`: you multiply by `z`, the input of ReLU. Its local gradient is its slope: 1 where `z` is above 0.",
                    nl="Je laatste regel geeft `{got}`: je vermenigvuldigt met `z`, de input van ReLU. Zijn local gradient is zijn helling: 1 waar `z` boven 0 ligt."),
            Mistake("w-for-x", wrong=_b(dw="dloss_dz * w"),
                    en="Your last line gives `{got}`: for `dloss_dw` you multiply by `w`. In `z = w * x + b`, `z` changes `x` times as fast as `w`: the local gradient for `w` is `x`.",
                    nl="Je laatste regel geeft `{got}`: voor `dloss_dw` vermenigvuldig je met `w`. In `z = w * x + b` verandert `z` `x` keer zo snel als `w`: de local gradient voor `w` is `x`."),
            Mistake("x-for-b", wrong=_b(db="dloss_dz * x"),
                    en="Your last line gives `{got}`: for `dloss_db` you multiply by `x`. `b` is added on its own, so the local gradient for `b` is 1.",
                    nl="Je laatste regel geeft `{got}`: voor `dloss_db` vermenigvuldig je met `x`. `b` wordt los opgeteld, dus de local gradient voor `b` is 1."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with one step of the backward pass: the slope of the line before it times one local gradient.",
                      "nl": "Vervang elke `...` door één stap van de backward pass: de helling van de regel ervoor keer één local gradient."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[dloss_dw, dloss_db]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[dloss_dw, dloss_db]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about `[-12, -6]`. Go back one node at a time: 2r, then −1, then 1 for ReLU where `z` is above 0, then `x` for `w` and 1 for `b`.",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer `[-12, -6]` zijn. Ga één node per keer terug: 2r, dan −1, dan 1 voor ReLU waar `z` boven 0 ligt, dan `x` voor `w` en 1 voor `b`."},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work each slope out from the values of the forward pass, and is `dloss_dz` 0 where `z` is 0 or below?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je elke helling uit met de waarden van de forward pass, en is `dloss_dz` 0 waar `z` 0 of lager is?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you work each slope out from the values of the forward pass?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Reken je elke helling uit met de waarden van de forward pass?"},
            "correct": {"en": "−12 and −6. Back from the loss: 6, −6, then −6 again through ReLU, because `z` = 4 is above 0, and then × 2 for `w` and × 1 for `b`. The check also tried other numbers, one of them with `z` below 0.",
                        "nl": "−12 en −6. Terug vanaf de loss: 6, −6, dan weer −6 door ReLU, omdat `z` = 4 boven 0 ligt, en dan × 2 voor `w` en × 1 voor `b`. De check probeerde ook andere getallen, waarvan één met `z` onder 0."},
        },
    ),
    "check": CodeBox(
        name="check",
        start=_C_START,
        solution=_C,
        compare=_close,
        keep=["def loss(w, b):", "return (y - (w * x + b)) ** 2", _C_END],
        first=[
            _DOTS,
            Mistake("typed", matches=lambda got, text: any(re.fullmatch(r"[-+*/().\s]*\d[-+*/().\d\s]*", rhs) for rhs in
                                                           re.findall(r"^\s*slope_[wb]\s*=([^#\n]*)", text, re.MULTILINE)),
                    en="You type a slope in yourself. Measure it with `loss`: the loss with one parameter moved by `h`, minus the loss at w = 1, b = 1, divided by `h`.",
                    nl="Je typt een helling zelf in. Meet hem met `loss`: de loss met één parameter verschoven met `h`, min de loss bij w = 1, b = 1, gedeeld door `h`."),
        ],
        mistakes=[
            Mistake("no-divide", wrong=_c(sw="loss(1 + h, 1) - loss(1, 1)", sb="loss(1, 1 + h) - loss(1, 1)"),
                    en="Your last line gives `{got}`: the change of the loss, not the slope. Divide each change by the step `h`.",
                    nl="Je laatste regel geeft `{got}`: de verandering van de loss, niet de helling. Deel elke verandering door de stap `h`."),
            Mistake("both", wrong=_c(sw="(loss(1 + h, 1 + h) - loss(1, 1)) / h", sb="(loss(1 + h, 1 + h) - loss(1, 1)) / h"),
                    en="Your last line gives `{got}`: you move `w` and `b` at the same time. A slope moves one parameter, and leaves the other at 1.",
                    nl="Je laatste regel geeft `{got}`: je verschuift `w` en `b` tegelijk. Een helling verschuift één parameter, en laat de andere op 1."),
            Mistake("swapped", wrong=_c(sw="(loss(1, 1 + h) - loss(1, 1)) / h", sb="(loss(1 + h, 1) - loss(1, 1)) / h"),
                    en="Your last line gives `{got}`: the two slopes are swapped. `loss(w, b)` takes `w` first: for `slope_w`, move the first number.",
                    nl="Je laatste regel geeft `{got}`: de twee hellingen zijn omgewisseld. `loss(w, b)` krijgt `w` eerst: verschuif voor `slope_w` het eerste getal."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...` with the slopes at w = 1, b = 1, measured by moving `w`, and then `b`, by `h`.",
                      "nl": "Vervang de twee `...` door de hellingen bij w = 1, b = 1, gemeten door `w`, en dan `b`, met `h` te verschuiven."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[slope_w, slope_b]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[slope_w, slope_b]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about `[-16, -4]`. Do you measure at w = 1 and b = 1, moving one parameter at a time by `h`, and divide by `h`?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer `[-16, -4]` zijn. Meet je bij w = 1 en b = 1, verschuif je één parameter per keer met `h`, en deel je door `h`?"},
            "correct": {"en": "About −16 and −4, as the backward pass gives, apart from a small difference because `h` is not 0. Moving needs a forward pass where you are and one per parameter: three here, but 100,001 with 100,000 parameters, so you check only a few slopes this way.",
                        "nl": "Ongeveer −16 en −4, zoals de backward pass geeft, op een klein verschil na omdat `h` geen 0 is. Verschuiven kost een forward pass waar je bent en één per parameter: hier drie, maar 100.001 met 100.000 parameters, dus zo controleer je maar een paar hellingen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("backward", _B_START, "empty", None),
    ("backward", _B, "correct", None),
    ("backward", _b(dy="-dloss_dr", dz="dloss_dy_pred", db="dloss_dz"), "correct", None),
    ("backward", _b(dy="dloss_dr"), "wrong", "sign"),
    ("backward", _b(dr="r"), "wrong", "no-two"),
    ("backward", _b(dz="dloss_dy_pred * z"), "wrong", "relu-value"),
    ("backward", _b(dw="dloss_dz * w"), "wrong", "w-for-x"),
    ("backward", _b(db="dloss_dz * x"), "wrong", "x-for-b"),
    ("backward", _b(dw="-12", db="-6"), "almost", "other-value"),
    ("backward", _B.replace("else:\n    dloss_dz = 0", "else:\n    dloss_dz = dloss_dy_pred"), "almost", "other-value"),
    ("check", _C_START, "empty", None),
    ("check", _C, "correct", None),
    ("check", _c(sw="(loss(1 + h, 1) - loss(1 - h, 1)) / (2 * h)", sb="(loss(1, 1 + h) - loss(1, 1 - h)) / (2 * h)"), "correct", None),
    ("check", _c(sw="-16", sb="-4"), "wrong", "typed"),
    ("check", _c(sw="-(16)", sb="2 * 2 * -1"), "wrong", "typed"),
    ("check", _c(sw="1 / h * (loss(1 + h, 1) - loss(1, 1))", sb="1 / h * (loss(1, 1 + h) - loss(1, 1))"), "correct", None),
    ("check", _c(sw="loss(1 + h, 1) - loss(1, 1)", sb="loss(1, 1 + h) - loss(1, 1)"), "wrong", "no-divide"),
    ("check", _c(sw="(loss(1 + h, 1 + h) - loss(1, 1)) / h", sb="(loss(1 + h, 1 + h) - loss(1, 1)) / h"), "wrong", "both"),
    ("check", _c(sw="(loss(1, 1 + h) - loss(1, 1)) / h", sb="(loss(1 + h, 1) - loss(1, 1)) / h"), "wrong", "swapped"),
    ("check", _c(sw="(loss(h, 0) - loss(0, 0)) / h", sb="(loss(0, h) - loss(0, 0)) / h"), "wrong", None),
]
