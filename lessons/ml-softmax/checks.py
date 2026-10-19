# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-softmax: softmax as a function, safe for big logits, and softmax with two classes, which
gives the sigmoid. And the picture of three probabilities as the first logit moves on the slider."""
import math
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

OTHERS = (1.0, 0.0)   # the logits of class 1 and class 2 in the picture; the slider sets class 0


def probabilities(z0: float) -> list[float]:
    """Softmax of (z0, 1, 0), with the largest logit taken off first."""
    zs = [z0, *OTHERS]
    top = max(zs)
    es = [math.exp(z - top) for z in zs]
    return [e / sum(es) for e in es]


def picture(z0: float, labels: dict[str, str], shown: list[str]) -> str:
    """Three bars, one per class, as high as their probability; `shown` are the probabilities written out, in the
    lesson's language. Beside them, the three probabilities for every z₀ of the slider, the line of class 0 solid, with
    a dashed line at 0.5 and a mark at z₀. Drawn as SVG text, once for every stop of the slider (L.explore); dragging
    across the second chart sets z₀."""
    from everything_ai import chart
    ps = probabilities(z0)
    c = chart.Panel((-0.6, 2.6), (0, 1.08), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xticks=[0, 1, 2])
    for i, p in enumerate(ps):
        c.rect(i - 0.3, 0, i + 0.3, p, "pred")
        c.text(i, p + 0.05, shown[i], "ink-2")
    zs = [-3 + 8 * i / 80 for i in range(81)]
    every = [probabilities(z) for z in zs]
    k = chart.Panel((-3, 5), (0, 1.08), title=labels["every"], xlabel="z₀", ylabel=labels["y"])
    k.line([-3, 5], [0.5, 0.5], "ink-3", 1.2, dashed=True)
    k.line([z0, z0], [0, 1.08], "ink-3", 1.2)
    for i in range(3):
        k.line(zs, [q[i] for q in every], "pred", 2.4 if i == 0 else 1.6, dashed=i > 0)
    k.points([z0] * 3, ps, "pred", 4.5)
    for i, y in ((0, every[-1][0] - 0.08), (1, every[0][1] + 0.07), (2, every[0][2] - 0.06)):   # between two stops: the mark at z₀ never crosses them
        k.text(4.75 if i == 0 else -2.75, y, str(i), "ink-2")
    return chart.figure([c, k], alt=labels["alt"], width=330, height=300, drag=1)   # two fit side by side in a play block


# The lesson is done when these are correct. "add-ten" is the choice question in the lesson.
REQUIRED = ["add-ten", "softmax", "two-classes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_S_HEAD = "import numpy as np\n\n\ndef softmax(z):\n    z = np.array(z)\n"
_S_START = _S_HEAD + "    ...\n"
_S = _S_HEAD + "    e = np.exp(z - z.max())\n    return e / e.sum()"


def _s(body: str) -> str:
    return _S_HEAD + body


_T_HEAD = "import numpy as np\n\nz = 1.5\nlogits = np.array([z, 0])\n"
_T_START = _T_HEAD + "p = ...\np\n"
_T = _T_HEAD + "p = np.exp(logits) / np.exp(logits).sum()\np"


def _t(new: str) -> str:
    return _T.replace("p = np.exp(logits) / np.exp(logits).sum()", new)


EXERCISES = {
    "softmax": FunctionBox(
        name="softmax",
        function="softmax",
        start=_S_START,
        solution=_S,
        visible=[([3, 1, 0],)],
        # [1000, 0, -1000]: taking the smallest logit off, or the mean, still gives np.exp(1000) or more there.
        hidden=[([2, 2],), ([0],), ([-1, 0, 1, 2],), ([1000, 999, 998],), ([1000, 0, -1000],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-exp", wrong=_s("    return z / z.sum()"),
                    en="`{call}` gives `{got}`: each logit divided by their sum. A logit of 0 then gets the probability 0, and a negative logit would get a negative number. Take `np.exp` of each logit first.",
                    nl="`{call}` geeft `{got}`: elke logit gedeeld door hun som. Een logit van 0 krijgt dan de kans 0, en een negatieve logit zou een negatief getal krijgen. Neem eerst `np.exp` van elke logit."),
            Mistake("no-divide", wrong=_s("    return np.exp(z - z.max())"),
                    en="`{call}` gives `{got}`: these do not add up to 1. Divide each by their sum: `e / e.sum()`.",
                    nl="`{call}` geeft `{got}`: die zijn samen geen 1. Deel elk door hun som: `e / e.sum()`."),
            Mistake("sum-of-logits", wrong=_s("    return np.exp(z) / z.sum()"),
                    en="`{call}` gives `{got}`: you divide by the sum of the logits. Divide by the sum of their powers, `np.exp(z).sum()`, so that the probabilities add up to 1.",
                    nl="`{call}` geeft `{got}`: je deelt door de som van de logits. Deel door de som van hun e-machten, `np.exp(z).sum()`, zodat de kansen samen 1 zijn."),
            Mistake("min-off", wrong=_s("    e = np.exp(z - z.min())\n    return e / e.sum()"),
                    en="`{call}` gives `{got}`: you take the smallest logit off, so the largest stays far above 0, and `np.exp` of it is too big for the computer. Take the largest logit off: `z.max()`. Then the biggest power is `np.exp(0)`, which is 1.",
                    nl="`{call}` geeft `{got}`: je haalt de kleinste logit eraf, dus de grootste blijft ver boven 0, en `np.exp` daarvan is te groot voor de computer. Haal de grootste logit eraf: `z.max()`. Dan is de grootste macht `np.exp(0)`, en dat is 1."),
            Mistake("overflow", matches=lambda got, text: _has_nan(got),
                    en="`{call}` gives `{got}`: `nan`, not a number. `np.exp(1000)` is too big for the computer and becomes `inf`, and `inf / inf` is `nan`. Subtract the largest logit first: `np.exp(z - z.max())`. That does not change the probabilities.",
                    nl="`{call}` geeft `{got}`: `nan`, not a number: geen getal. `np.exp(1000)` is te groot voor de computer en wordt `inf`, en `inf / inf` is `nan`. Trek eerst de grootste logit af: `np.exp(z - z.max())`. Aan de kansen verandert dat niets."),
            Mistake("too-big", error=lambda e: isinstance(e, OverflowError),
                    en="With `{call}`, your function stops: e to the power of a big logit is too big for the computer. Subtract the largest logit first, as in step 1 of the recipe. Then no power gets bigger than 1.",
                    nl="Met `{call}` stopt je functie: e tot de macht van een grote logit is te groot voor de computer. Trek eerst de grootste logit af, zoals in stap 1 van het recept. Dan wordt geen macht groter dan 1."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with steps 1 to 3 of the recipe: take the largest logit off, take `np.exp`, and divide by the sum.",
                      "nl": "Vervang de `...` door stap 1 tot en met 3 van het recept: haal de grootste logit eraf, neem `np.exp`, en deel door de som."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of logits, and the `def` line of your function has a different number of names between its parentheses. Keep `(z)`.",
                          "nl": "De check roept `{call}` aan, met één lijst logits, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(z)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is each probability `np.exp` of its logit, divided by the sum of all of those?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is elke kans `np.exp` van zijn logit, gedeeld door de som van al die e-machten?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of logits, also very big ones?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk aantal logits, ook heel grote?"},
            "correct": {"en": "Your function is right in all {n} cases. The logits 1000, 999 and 998 give the same probabilities as 2, 1 and 0: only the differences count, and taking the largest off keeps `np.exp` small.",
                        "nl": "Je functie klopt in alle {n} gevallen. De logits 1000, 999 en 998 geven dezelfde kansen als 2, 1 en 0: alleen de verschillen tellen, en de grootste eraf halen houdt `np.exp` klein."},
        },
    ),
    "two-classes": CodeBox(
        name="two-classes",
        start=_T_START,
        solution=_T,
        keep=["logits = np.array([z, 0])"],
        # The check also runs the box with another logit: a number typed in by hand does not change with it.
        others=[{"z": -2}],
        first=[_DOTS],
        mistakes=[
            Mistake("other-box", error=lambda e: isinstance(e, NameError) and "'softmax'" in str(e),
                    en="Each box runs on its own, so the function `softmax` from exercise 2 does not exist in this one. Work it out here: `np.exp` of each logit, divided by the sum of both.",
                    nl="Elk vak draait op zichzelf, dus de functie `softmax` uit oefening 2 bestaat in dit vak niet. Reken het hier uit: `np.exp` van elke logit, gedeeld door de som van beide."),
            Mistake("sigmoid-each", wrong=_t("p = 1 / (1 + np.exp(-logits))"),
                    en="Your last line gives `{got}`: the sigmoid of each logit on its own. Those do not add up to 1. Softmax takes `np.exp` of each and divides by their sum.",
                    nl="Je laatste regel geeft `{got}`: de sigmoid van elke logit apart. Die zijn samen geen 1. Softmax neemt `np.exp` van elke logit en deelt door hun som."),
            Mistake("no-divide", wrong=_t("p = np.exp(logits)"),
                    en="Your last line gives `{got}`: these do not add up to 1. Divide by their sum.",
                    nl="Je laatste regel geeft `{got}`: die zijn samen geen 1. Deel door hun som."),
            Mistake("one-number", matches=lambda got, text: isinstance(got, (int, float)) and not isinstance(got, bool),
                    en="Your last line gives one number. The task asks for both probabilities, as an array: one per class.",
                    nl="Je laatste regel geeft één getal. De opdracht vraagt beide kansen, als array: één per klasse."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the softmax of `logits`: two probabilities.",
                      "nl": "Vervang de `...` door de softmax van `logits`: twee kansen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another logit by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere logit."},
            "no_value": {"en": "The last line of the box shows nothing. Put `p` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `p` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is it `np.exp` of each logit, divided by the sum of both?",
                      "nl": "Je laatste regel geeft `{got}`. Is het `np.exp` van elke logit, gedeeld door de som van beide?"},
            "other": {"en": "Right for z = 1.5. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `logits` itself?",
                      "nl": "Goed voor z = 1,5. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `logits` zelf?"},
            "other_error": {"en": "Right for z = 1.5. But with {where}, your code stops with an error. Do you work it out from `logits` itself?",
                            "nl": "Goed voor z = 1,5. Maar met {where} stopt je code met een fout. Reken je het uit met `logits` zelf?"},
            "correct": {"en": "The first probability is about 0.818: exactly the sigmoid of 1.5. With two classes and the second logit 0, softmax is the sigmoid. The check also tried z = −2: 0.119, the sigmoid of −2.",
                        "nl": "De eerste kans is ongeveer 0,818: precies de sigmoid van 1,5. Met twee klassen en de tweede logit 0 is softmax de sigmoid. De check probeerde ook z = −2: 0,119, de sigmoid van −2."},
        },
    ),
}


def _has_nan(got) -> bool:
    try:
        return any(math.isnan(float(v)) for v in got)
    except (TypeError, ValueError):
        return False


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("softmax", _S_START, "empty", None),
    ("softmax", _S, "correct", None),
    ("softmax", _s("    e = np.exp(z - np.max(z))\n    return e / np.sum(e)"), "correct", None),
    ("softmax", _s("    z = z - z.max()\n    return np.exp(z) / np.exp(z).sum()"), "correct", None),
    ("softmax", _s("    return z / z.sum()"), "wrong", "no-exp"),
    ("softmax", _s("    return np.exp(z - z.max())"), "wrong", "no-divide"),
    ("softmax", _s("    return np.exp(z) / z.sum()"), "wrong", "sum-of-logits"),
    ("softmax", _s("    return np.exp(z) / np.exp(z).sum()"), "wrong", "overflow"),
    ("softmax", _s("    e = np.exp(z - z.min())\n    return e / e.sum()"), "wrong", "min-off"),
    ("softmax", _s("    e = np.exp(z - z.mean())\n    return e / e.sum()"), "wrong", "overflow"),
    ("softmax", "import math\n\n\ndef softmax(z):\n    es = []\n    for x in z:\n        es.append(math.exp(x))\n    return [e / sum(es) for e in es]", "error", "too-big"),
    ("softmax", "import math\n\n\ndef softmax(z):\n    top = max(z)\n    es = []\n    for x in z:\n        es.append(math.exp(x - top))\n    total = sum(es)\n    ps = []\n    for e in es:\n        ps.append(e / total)\n    return ps", "correct", None),
    ("two-classes", _T_START, "empty", None),
    ("two-classes", _T, "correct", None),
    ("two-classes", _t("e = np.exp(logits - logits.max())\np = e / e.sum()"), "correct", None),
    ("two-classes", _t("p = 1 / (1 + np.exp(-logits))"), "wrong", "sigmoid-each"),
    ("two-classes", _t("p = np.exp(logits)"), "wrong", "no-divide"),
    ("two-classes", _t("p = np.exp(z) / np.exp(logits).sum()"), "wrong", "one-number"),
    ("two-classes", _t("p = softmax(logits)"), "error", "other-box"),
    ("two-classes", _t("s = 1 / (1 + np.exp(-z))\np = np.array([s, 1 - s])"), "correct", None),
    ("two-classes", _t("p = np.array([0.8175744761936437, 0.18242552380635635])"), "almost", "other-value"),
]
