# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-validation-set: choosing the degree of a curve by its MSE on the validation set. And the
picture of the training and validation MSE of every degree for the temperatures of the worked example."""
import re

from everything_ai.check import Code, CodeBox, Mistake

SEED, N, DEGREES = 75, 32, range(1, 10)   # as in the worked example: 12 to train, 10 to validate, 10 to test


def _split():
    import numpy as np
    rng = np.random.default_rng(SEED)
    hours = 24 * rng.random(N)
    temps = 20 - 0.05 * (hours - 15) ** 2 + 3 * (rng.random(N) - 0.5)
    order = rng.permutation(N)
    return {name: (hours[idx], temps[idx]) for name, idx in
            (("train", order[:12]), ("val", order[12:22]), ("test", order[22:]))}


SETS = _split()


def _mse(w, name: str) -> float:
    import numpy as np
    x, y = SETS[name]
    return float(np.mean((y - np.polyval(w, x)) ** 2))


def _errors():
    import numpy as np
    out = {}
    for d in DEGREES:
        w = np.polyfit(*SETS["train"], d)
        out[d] = {name: _mse(w, name) for name in SETS}
    return out


ERRORS = _errors()
BEST = min(DEGREES, key=lambda d: ERRORS[d]["val"])
TOP = 15   # the MSE axis stops here, with ticks 0, 5, 10 and 15; a higher validation MSE gets its number at the top


def picture(labels: dict[str, str], shown: dict[int, str]) -> str:
    """The training MSE (filled dots) and the validation MSE (open dots) of every degree, the chosen degree larger.
    `shown` writes out the validation MSEs above the axis, in the lesson's language."""
    from everything_ai import chart
    ds = list(DEGREES)
    train = [ERRORS[d]["train"] for d in ds]
    val = [min(ERRORS[d]["val"], TOP + 1) for d in ds]
    c = chart.Panel((0.5, 9.5), (0, TOP), title=labels["title"], xlabel=labels["degree"], ylabel=labels["mse"], xticks=ds)
    c.line(ds, train, "loss", 2.0)
    c.line(ds, val, "loss", 2.0, dashed=True)
    c.points(ds, train, "loss", 3.5)
    c.points(ds, val, "loss", 3.5, hollow=True)
    c.points([BEST], [ERRORS[BEST]["val"]], "loss", 7, hollow=True)
    c.text(BEST + 0.6, ERRORS[BEST]["val"] + 1.4, labels["chosen"], "ink")   # right of the dot, clear of the lines from degree 1
    for d in ds:
        if ERRORS[d]["val"] > TOP:
            c.text(d, TOP - 0.8, f"{shown[d]} ↑", "ink-2")
    return chart.figure([c], alt=labels["alt"], width=620, height=300)


# The lesson is done when these are correct. "test-to-choose" and "test-at-end" are the choice questions.
REQUIRED = ["test-to-choose", "choose-degree", "test-at-end"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_C_HEAD = ("import numpy as np\n\nx_train = np.array([8.4, 23.9, 6.0, 22.7, 5.1, 20.1, 22.4, 4.2, 9.9, 0.5, 20.8, 23.6])\n"
           "y_train = np.array([17.8, 15.2, 15.5, 15.5, 14.1, 19.4, 18.7, 15.5, 18.6, 8.8, 18.4, 17.3])\n"
           "x_val = np.array([10.3, 4.9, 16.0, 2.2, 16.4, 21.8])\ny_val = np.array([19.3, 16.2, 20.7, 12.4, 19.7, 16.3])\n"
           "degrees = [1, 2, 3, 4, 5, 6]\nval_mses = []\nfor degree in degrees:\n")
_C_START = _C_HEAD + "    ...\nbest = ...\nbest\n"
_C_BODY = ("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
           "best = degrees[np.argmin(val_mses)]\n")
_C = _C_HEAD + _C_BODY + "best"


def _c(body: str) -> str:
    return _C_HEAD + body + "best"


def _position(got, text) -> bool:
    return re.search(r"best\s*=\s*(np\.argmin\(|np\.array\(val_mses\)\.argmin\(|val_mses\.index\()", text) is not None


# Fitting on the validation set gives the same outcome as a right or a known wrong answer, so it is known by the text.
_FIT_ON_VAL = Mistake("fit-on-val", matches=lambda got, text: re.search(r"^[^#\n]*polyfit\(\s*x_val", text, re.MULTILINE) is not None,
                      en="You let `np.polyfit` fit the curves on `x_val` and `y_val`, the validation set. Fit them on the training set, `x_train` and `y_train`, and measure them on `x_val` and `y_val`: examples the curves did not train on.",
                      nl="Je laat `np.polyfit` de curves aanpassen op `x_val` en `y_val`, de validation set. Pas ze aan op de training set, `x_train` en `y_train`, en meet ze op `x_val` en `y_val`: voorbeelden waarop de curves niet trainden.")


EXERCISES = {
    "choose-degree": CodeBox(
        name="choose-degree",
        start=_C_START,
        solution=_C,
        keep=["for degree in degrees:"],
        # The check also runs the box with other degrees to choose from: a degree typed in by hand does not change with them.
        others=[{"degrees": Code("[1, 3, 5]")}],
        first=[_DOTS],
        # Fitted on the validation set and measured on the training set gives degree 2 as well: wrong all the same.
        form=[_FIT_ON_VAL],
        mistakes=[
            Mistake("list-argmin", error=lambda e: isinstance(e, AttributeError) and "'list' object has no attribute 'argmin'" in str(e),
                    en="A list has no `.argmin()`; that belongs to a NumPy array. `np.argmin(val_mses)` takes the list as it is, or make an array of it first: `np.array(val_mses).argmin()`.",
                    nl="Een lijst heeft geen `.argmin()`; die hoort bij een NumPy-array. `np.argmin(val_mses)` neemt de lijst zoals hij is, of maak er eerst een array van: `np.array(val_mses).argmin()`."),
            _FIT_ON_VAL,
            Mistake("highest", matches=lambda got, text: "argmax" in text,
                    en="`np.argmax` gives the position of the highest validation MSE: the worst degree. The best one has the lowest: `np.argmin`.",
                    nl="`np.argmax` geeft de positie van de hoogste validation-MSE: de slechtste graad. De beste heeft de laagste: `np.argmin`."),
            Mistake("position", matches=_position,
                    en="Your last line gives `{got}`: the position of the lowest validation MSE in the list, counting from 0, not the degree itself. Look up the degree at that position: `degrees[...]`.",
                    nl="Je laatste regel geeft `{got}`: de positie van de laagste validation-MSE in de lijst, geteld vanaf 0, niet de graad zelf. Zoek de graad op die positie op: `degrees[...]`."),
            Mistake("on-training", wrong=_c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_train - np.polyval(w, x_train)) ** 2))\n"
                                           "best = degrees[np.argmin(val_mses)]\n"),
                    en="Your last line gives `{got}`, the highest degree in the list: the one the training MSE chooses, because on the training set more capacity always looks better. Do you measure each curve on `x_val` and `y_val`?",
                    nl="Je laatste regel geeft `{got}`, de hoogste graad in de lijst: die kiest de training-MSE, want op de training set ziet meer capaciteit er altijd beter uit. Meet je elke curve op `x_val` en `y_val`?"),
            Mistake("lowest-mse", wrong=_c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                                          "best = min(val_mses)\n"),
                    en="Your last line gives the lowest validation MSE itself, about 0.96, not the degree that has it. At which position in `val_mses` is it, and which degree is at that position in `degrees`?",
                    nl="Je laatste regel geeft de laagste validation-MSE zelf, ongeveer 0,96, niet de graad die erbij hoort. Op welke positie in `val_mses` staat hij, en welke graad staat op die positie in `degrees`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in the loop, train a curve of each degree and add its validation MSE to `val_mses`; then put the degree with the lowest validation MSE in `best`.",
                      "nl": "Vervang de twee `...`: train in de loop een curve van elke graad en voeg zijn validation-MSE toe aan `val_mses`; zet daarna de graad met de laagste validation-MSE in `best`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other degrees by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere graden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `best` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `best` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you fit each curve on the training set, measure it on the validation set, and take the degree with the lowest validation MSE?",
                      "nl": "Je laatste regel geeft `{got}`. Pas je elke curve aan op de training set, meet je hem op de validation set, en neem je de graad met de laagste validation-MSE?"},
            "other": {"en": "Right for these degrees. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you look up the best degree in `degrees` itself?",
                      "nl": "Goed voor deze graden. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Zoek je de beste graad op in `degrees` zelf?"},
            "other_error": {"en": "Right for these degrees. But with {where}, your code stops with an error. Do you look up the best degree in `degrees` itself?",
                            "nl": "Goed voor deze graden. Maar met {where} stopt je code met een fout. Zoek je de beste graad op in `degrees` zelf?"},
            "correct": {"en": "Degree 2, with a validation MSE of about 0.96. The training MSE would have chosen degree 6. The check also tried other degrees.",
                        "nl": "Graad 2, met een validation-MSE van ongeveer 0,96. De training-MSE had graad 6 gekozen. De check probeerde ook andere graden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("choose-degree", _C_START, "empty", None),
    ("choose-degree", _C, "correct", None),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    pred = np.polyval(w, x_val)\n    val_mses.append(((y_val - pred) ** 2).mean())\n"
                         "best = degrees[val_mses.index(min(val_mses))]\n"), "correct", None),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = degrees[np.argmax(val_mses)]\n"), "wrong", "highest"),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = np.argmin(val_mses)\n"), "wrong", "position"),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_train - np.polyval(w, x_train)) ** 2))\n"
                         "best = degrees[np.argmin(val_mses)]\n"), "wrong", "on-training"),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = 2\n"), "almost", "other-value"),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = degrees[np.array(val_mses).argmin()]\n"), "correct", None),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    mse = np.mean((y_val - np.polyval(w, x_val)) ** 2)\n    val_mses.append(mse)\n"
                         "    if mse <= min(val_mses):\n        best = degree\n"), "correct", None),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = degrees[val_mses.argmin()]\n"), "error", "list-argmin"),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = val_mses.index(min(val_mses))\n"), "wrong", "position"),
    ("choose-degree", _c("    w = np.polyfit(x_val, y_val, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = degrees[np.argmin(val_mses)]\n"), "wrong", "fit-on-val"),
    ("choose-degree", _c("    w = np.polyfit(x_val, y_val, degree)\n    val_mses.append(np.mean((y_train - np.polyval(w, x_train)) ** 2))\n"
                         "best = degrees[np.argmin(val_mses)]\n"), "wrong", "fit-on-val"),
    ("choose-degree", _c("    w = np.polyfit(x_train, y_train, degree)\n    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
                         "best = min(val_mses)\n"), "wrong", "lowest-mse"),
]
