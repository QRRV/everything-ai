# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-softmax-gradient: the gradient of the cross-entropy for the logits, the probabilities minus
the one-hot label, as a function; and the slope for one logit measured by nudging it. And the picture of the
probabilities and the gradient as the logit of the right class moves on the slider."""
import math
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, same_value

OTHERS = (2.0, 0.0)   # the logits of class 0 (cat) and class 2 (bird) in the picture; the slider sets class 1 (dog)
RIGHT = 1             # the right class in the picture: the photo is of a dog


def probabilities(z_right: float) -> list[float]:
    """Softmax of (2, z_right, 0), with the largest logit taken off first."""
    zs = [OTHERS[0], z_right, OTHERS[1]]
    top = max(zs)
    es = [math.exp(z - top) for z in zs]
    return [e / sum(es) for e in es]


def gradient(z_right: float) -> list[float]:
    """The gradient of the cross-entropy for the three logits: the probabilities minus the one-hot label of dog."""
    return [p - (1 if k == RIGHT else 0) for k, p in enumerate(probabilities(z_right))]


def picture(z_right: float, labels: dict, shown_p: list[str]) -> str:
    """Three bars as high as the probabilities, the right class's drawn heavier; beside them, the slope for each logit
    for every logit of dog on the slider, dog's line the heavier one, with dashed lines at -0.1 and 0.1 and a dot at
    each slope for `z_right`. `shown_p` are the probabilities written out, in the lesson's language. Drawn as SVG
    text, once for every stop of the slider (L.explore); dragging across the second chart sets the logit of dog."""
    from everything_ai import chart
    ps, gs = probabilities(z_right), gradient(z_right)
    bars = chart.Panel((-0.6, 2.6), (0, 1.12), title=labels["bars"], xlabel=labels["class"], ylabel=labels["p"],
                       xticks=[0, 1, 2])
    for i, p in enumerate(ps):
        bars.rect(i - 0.3, 0, i + 0.3, p, "pred")
        if i == RIGHT:   # the right class: a second outline, so it stands out without another colour
            bars.line([i - 0.3, i - 0.3, i + 0.3, i + 0.3], [0, p, p, 0], "pred", 3.2)
        bars.text(i, p + 0.05, shown_p[i], "ink" if i == RIGHT else "ink-2")
    zs = [-3 + 8 * i / 80 for i in range(81)]
    every = [gradient(z) for z in zs]
    slopes = chart.Panel((-3, 5), (-1.25, 1.12), title=labels["slopes"], xlabel="z₁", ylabel=labels["slope"])
    slopes.line([-3, 5], [0, 0], "ink-3", 1.2)
    for y in (-0.1, 0.1):   # the task: from where are all three slopes within 0.1 of 0
        slopes.line([-3, 5], [y, y], "ink-3", 1.2, dashed=True)
    slopes.line([z_right, z_right], [-1.25, 1.12], "ink-3", 1.2)
    for k in range(3):
        slopes.line(zs, [g[k] for g in every], "grad", 3.2 if k == RIGHT else 1.6)
    slopes.points([z_right] * 3, gs, "grad", 4.5)
    for k, y in ((0, every[8][0] + 0.12), (RIGHT, every[8][RIGHT] - 0.14), (2, every[8][2] + 0.12)):   # every[8] is at z₁ = -2.2
        slopes.text(-2.2, y, labels["names"][k], "ink" if k == RIGHT else "ink-2")
    return chart.figure([bars, slopes], alt=labels["alt"], width=330, height=300, drag=1)   # two fit side by side in a play block


# The lesson is done when these are correct. "one-step" is the choice question in the lesson.
REQUIRED = ["one-step", "logit-gradient", "nudge"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _number(x: float) -> str:
    text = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _rounded(v):
    """Numbers to six decimals, so that an answer rounded to the four of the lesson never shows the same as the right
    one, and an array as a list; anything else the way Python shows it."""
    import numpy as np
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return _number(float(a))
    if a.ndim == 1:
        return "[" + ", ".join(_number(float(x)) for x in a) + "]"
    return repr(v)


def _flat(v):
    """One row or one column of numbers as a plain row: the lesson is not about shapes (template 4.3)."""
    import numpy as np
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return v
    return a.ravel() if a.ndim == 2 and 1 in a.shape else v


_G_HEAD = ("import numpy as np\n\n\ndef softmax(z):\n    e = np.exp(np.array(z) - max(z))\n    return e / e.sum()\n\n\n"
           "def logit_gradient(z, label):\n")
_G_START = _G_HEAD + "    ...\n"
_G = _G_HEAD + "    return softmax(z) - np.eye(len(z))[label]"


def _g(body: str) -> str:
    return _G_HEAD + "    return " + body


_N_HEAD = ("import numpy as np\n\n\ndef loss(z, label):\n    e = np.exp(z - z.max())\n    return -np.log(e[label] / e.sum())\n\n\n"
           "z = np.array([0.5, 2.0, -1.0])\nlabel = 0\nh = 0.0001\n")
_N_START = _N_HEAD + "slope = ...\nslope\n"
_N_LINE = "slope = (loss(z + np.array([h, 0, 0]), label) - loss(z, label)) / h"
_N = _N_HEAD + _N_LINE + "\nslope"


def _n(new: str) -> str:
    return _N.replace(_N_LINE, new)


EXERCISES = {
    "logit-gradient": FunctionBox(
        name="logit-gradient",
        function="logit_gradient",
        start=_G_START,
        solution=_G,
        visible=[([2, 1, 0], 1)],
        hidden=[([0, 0], 0), ([3, 1, 0, -1], 3), ([0.5, 2.0, -1.0], 2), ([1000, 999, 998], 0), ([5], 0)],
        show=lambda v: _rounded(_flat(v)),
        compare=lambda got, want: same_value(_flat(got), want),
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_g("np.eye(len(z))[label] - softmax(z)"),
                    en="`{call}` gives `{got}`: the right size, the wrong sign. The gradient is the probabilities minus the one-hot label, so the right class gets a negative slope.",
                    nl="`{call}` geeft `{got}`: de goede grootte, het verkeerde teken. De gradient is de kansen min het one-hot-label, dus de goede klasse krijgt een negatieve helling."),
            Mistake("no-one-hot", wrong=_g("softmax(z)"),
                    en="`{call}` gives `{got}`: the probabilities alone. Take the one-hot label off them: the right class loses 1.",
                    nl="`{call}` geeft `{got}`: alleen de kansen. Haal het one-hot-label ervan af: de goede klasse verliest 1."),
            Mistake("minus-one", wrong=_g("softmax(z) - 1"),
                    en="`{call}` gives `{got}`: you take 1 off every class. Only the right class has a 1 in its one-hot label: `np.eye(len(z))[label]`.",
                    nl="`{call}` geeft `{got}`: je haalt 1 van elke klasse af. Alleen de goede klasse heeft een 1 in het one-hot-label: `np.eye(len(z))[label]`."),
            Mistake("logits", wrong=_g("np.array(z) - np.eye(len(z))[label]"),
                    en="`{call}` gives `{got}`: the logits minus the one-hot label. The gradient takes the probabilities: put the logits through `softmax` first.",
                    nl="`{call}` geeft `{got}`: de logits min het one-hot-label. De gradient neemt de kansen: haal de logits eerst door `softmax`."),
            Mistake("fixed-length", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="With `{call}`, your function stops: the one-hot label and the probabilities do not have the same length. Make the one-hot label as long as `z`, for any number of classes: `np.eye(len(z))[label]`.",
                    nl="Met `{call}` stopt je functie: het one-hot-label en de kansen zijn niet even lang. Maak het one-hot-label zo lang als `z`, voor elk aantal klassen: `np.eye(len(z))[label]`."),
            Mistake("right-class-only", wrong=_G_HEAD + "    g = np.zeros(len(z))\n    g[label] = softmax(z)[label] - 1\n    return g",
                    en="`{call}` gives `{got}`: a slope for the right class only. The other logits have a slope too: softmax shares out 1, so more for a wrong class leaves less for the right one. They get their own probability.",
                    nl="`{call}` geeft `{got}`: alleen een helling voor de goede klasse. De andere logits hebben ook een helling: softmax verdeelt 1, dus meer voor een foute klasse laat minder over voor de goede. Zij krijgen hun eigen kans."),
            Mistake("right-only", wrong=_g("softmax(z)[label] - 1"),
                    en="`{call}` gives `{got}`: one number, the slope for the logit of the right class. Every logit has a slope: give back one per class.",
                    nl="`{call}` geeft `{got}`: één getal, de helling voor de logit van de goede klasse. Elke logit heeft een helling: geef er één per klasse terug."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `return` line: the probabilities from `softmax(z)` minus the one-hot label.",
                      "nl": "Vervang de `...` door een `return`-regel: de kansen uit `softmax(z)` min het one-hot-label."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the logits and the label, and the `def` line of your function has a different number of names between its parentheses. Keep `(z, label)`.",
                          "nl": "De check roept `{call}` aan, met de logits en het label, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(z, label)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is it the probabilities from `softmax(z)` minus the one-hot label of `label`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is het de kansen uit `softmax(z)` min het one-hot-label van `label`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of classes and any label?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk aantal klassen en elk label?"},
            "correct": {"en": "Your function is right in all {n} cases. With one class the gradient is 0: softmax always gives it 1, so nothing can change.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met één klasse is de gradient 0: softmax geeft hem altijd 1, dus er kan niets veranderen."},
        },
    ),
    "nudge": CodeBox(
        name="nudge",
        start=_N_START,
        solution=_N,
        keep=["label = 0"],
        # The check also runs the box with other logits: a number typed in by hand does not change with them.
        others=[{"z": Code("np.array([1.0, -0.5, 0.0])")}],
        # A nudge measures the slope to about 4 decimals: another small h, or a nudge both ways, is right too.
        compare=lambda got, want: same_value(got, want, rel=0.001),
        first=[_DOTS],
        mistakes=[
            Mistake("one-value", error=lambda e: isinstance(e, TypeError) and "required positional argument: 'label'" in str(e),
                    en="In the box, `loss` needs two values, `loss(z, label)`, and your code gives it one. The worked example's `_loss` had only the logits; here the label goes in too.",
                    nl="In het vak heeft `loss` twee waarden nodig, `loss(z, label)`, en je code geeft er één. De `_loss` van het uitgewerkte voorbeeld had alleen de logits; hier gaat het label er ook in."),
            Mistake("no-divide", wrong=_n("slope = loss(z + np.array([h, 0, 0]), label) - loss(z, label)"),
                    en="Your last line gives a tiny number: how much the loss changes, not its slope. Divide by `h`, the size of the step.",
                    nl="Je laatste regel geeft een heel klein getal: hoeveel de loss verandert, niet zijn helling. Deel door `h`, de grootte van het stapje."),
            Mistake("brackets", wrong=_n("slope = loss(z + np.array([h, 0, 0]), label) - loss(z, label) / h"),
                    en="Your last line gives `{got}`. Without brackets, Python divides only `loss(z, label)` by `h`. Put the whole difference between brackets.",
                    nl="Je laatste regel geeft `{got}`. Zonder haakjes deelt Python alleen `loss(z, label)` door `h`. Zet het hele verschil tussen haakjes."),
            Mistake("all-logits", wrong=_n("slope = (loss(z + h, label) - loss(z, label)) / h"),
                    en="Your last line gives about 0: both losses come out the same. With `z + h`, every logit changes by the tiny step, and softmax only looks at the differences between logits. Change only the logit of class 0, in a new array: `z + np.array([h, 0, 0])`.",
                    nl="Je laatste regel geeft ongeveer 0: beide losses komen hetzelfde uit. Met `z + h` verandert elke logit met het kleine stapje, en softmax kijkt alleen naar de verschillen tussen logits. Verander alleen de logit van klasse 0, in een nieuwe array: `z + np.array([h, 0, 0])`."),
            Mistake("other-logit", wrong=_n("slope = (loss(z + np.array([0, h, 0]), label) - loss(z, label)) / h"),
                    en="Your last line gives `{got}`: the slope for the logit of class 1. Change the logit of class 0, the first place: `np.array([h, 0, 0])`.",
                    nl="Je laatste regel geeft `{got}`: de helling voor de logit van klasse 1. Verander de logit van klasse 0, de eerste plek: `np.array([h, 0, 0])`."),
            Mistake("backwards", wrong=_n("slope = (loss(z, label) - loss(z + np.array([h, 0, 0]), label)) / h"),
                    en="Your last line gives `{got}`: the right size, the wrong sign. The slope is the loss after the step minus the loss before, divided by `h`.",
                    nl="Je laatste regel geeft `{got}`: de goede grootte, het verkeerde teken. De helling is de loss na het stapje min de loss ervoor, gedeeld door `h`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope: the loss with the logit of class 0 changed by the tiny step `h`, minus the loss without, divided by `h`.",
                      "nl": "Vervang de `...` door de helling: de loss met de logit van klasse 0 een heel klein stapje `h` veranderd, min de loss zonder, gedeeld door `h`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other logits by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere logits."},
            "no_value": {"en": "The last line of the box shows nothing. Put `slope` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `slope` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you work out the loss with only the logit of class 0 changed by `h`, take the loss of `z` itself off, and divide by `h`? And is `h` still tiny, such as 0.0001?",
                      "nl": "Je laatste regel geeft `{got}`. Reken je de loss uit met alleen de logit van klasse 0 veranderd met `h`, haal je de loss van `z` zelf eraf, en deel je door `h`? En is `h` nog heel klein, zoals 0,0001?"},
            "other": {"en": "Right for these logits. But with {where}, your last line gives `{got}`, and it should give about `-0.3715`. Do you work it out with `loss`, `z` and `h` themselves?",
                      "nl": "Goed voor deze logits. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `-0.3715` zijn. Reken je het uit met `loss`, `z` en `h` zelf?"},
            "other_error": {"en": "Right for these logits. But with {where}, your code stops with an error. Do you work it out with `loss`, `z` and `h` themselves?",
                            "nl": "Goed voor deze logits. Maar met {where} stopt je code met een fout. Reken je het uit met `loss`, `z` en `h` zelf?"},
            "correct": {"en": "The slope is about −0.82: the probability of class 0, about 0.18, minus 1, as the formula says. Negative, so gradient descent raises the logit of the right class. The check also tried other logits.",
                        "nl": "De helling is ongeveer −0,82: de kans op klasse 0, ongeveer 0,18, min 1, zoals de formule zegt. Negatief, dus gradient descent verhoogt de logit van de goede klasse. De check probeerde ook andere logits."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("logit-gradient", _G_START, "empty", None),
    ("logit-gradient", _G, "correct", None),
    ("logit-gradient", _G_HEAD + "    p = softmax(z)\n    p[label] = p[label] - 1\n    return p", "correct", None),
    ("logit-gradient", _G_HEAD + "    onehot = np.zeros(len(z))\n    onehot[label] = 1\n    return softmax(z) - onehot", "correct", None),
    ("logit-gradient", _G_HEAD + "    return list(softmax(z) - np.eye(len(z))[label])", "correct", None),
    ("logit-gradient", _g("np.eye(len(z))[label] - softmax(z)"), "wrong", "sign"),
    ("logit-gradient", _g("softmax(z)"), "wrong", "no-one-hot"),
    ("logit-gradient", _g("softmax(z) - 1"), "wrong", "minus-one"),
    ("logit-gradient", _g("np.array(z) - np.eye(len(z))[label]"), "wrong", "logits"),
    ("logit-gradient", _g("softmax(z)[label] - 1"), "wrong", "right-only"),
    ("logit-gradient", _g("softmax(z) - np.eye(3)[label]"), "error", "fixed-length"),
    ("logit-gradient", _g("softmax(z) - np.eye(len(z))[[label]]"), "correct", None),
    ("logit-gradient", _G_HEAD + "    g = np.zeros(len(z))\n    g[label] = softmax(z)[label] - 1\n    return g", "wrong", "right-class-only"),
    ("logit-gradient", _g("[0.6652, -0.7553, 0.0900]"), "wrong", None),
    ("nudge", _N_START, "empty", None),
    ("nudge", _N, "correct", None),
    ("nudge", _n("slope = (loss(z + np.eye(3)[label] * h, label) - loss(z, label)) / h"), "correct", None),
    ("nudge", _n("slope = (loss(z + np.array([h, 0, 0]), label) - loss(z - np.array([h, 0, 0]), label)) / (2 * h)"), "correct", None),
    ("nudge", _n("e = np.exp(z - z.max())\nslope = e[0] / e.sum() - 1"), "correct", None),
    ("nudge", _n("slope = loss(z + np.array([h, 0, 0]), label) - loss(z, label)"), "wrong", "no-divide"),
    ("nudge", _n("slope = (loss(z + h, label) - loss(z, label)) / h"), "wrong", "all-logits"),
    ("nudge", _n("slope = (loss(z + np.array([0, h, 0]), label) - loss(z, label)) / h"), "wrong", "other-logit"),
    ("nudge", _n("slope = (loss(z, label) - loss(z + np.array([h, 0, 0]), label)) / h"), "wrong", "backwards"),
    ("nudge", _n("slope = -0.8247"), "almost", "other-value"),
    ("nudge", _n("slope = (loss(z + np.array([h, 0, 0])) - loss(z)) / h"), "error", "one-value"),
    ("nudge", _n("slope = loss(z + np.array([h, 0, 0]), label) - loss(z, label) / h"), "wrong", "brackets"),
    ("nudge", _n("z2 = z\nz2[0] = z2[0] + h\nslope = (loss(z2, label) - loss(z, label)) / h"), "wrong", "all-logits"),
]
