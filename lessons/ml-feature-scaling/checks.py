# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-feature-scaling: standardize the features of four students, and predict the price of a
new house with a model trained on standardized features. And the picture: the loss landscape of one feature, in m²
and standardized, each with 50 steps of gradient descent."""
import re

from everything_ai.check import Code, CodeBox, Mistake

# The houses of the previous lesson, in their raw units: area in m², a garden (1 = yes), distance to the station in m.
HOUSES_RAW = [[62, 0, 400], [85, 1, 1200], [110, 1, 800], [48, 0, 300],
              [95, 0, 1500], [130, 1, 600], [73, 1, 900], [58, 0, 1100]]
PRICES = [191, 218, 290, 162, 215, 333, 211, 150]

# One feature for the picture: the area in m², and the prices of the houses in the gradient descent lessons.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE_1 = [245, 310, 395, 198, 320, 465, 280, 232]


def _path(x, lr, steps):
    """Where (w, b) is after each step of gradient descent from (0, 0), for the model w · x + b."""
    import numpy as np
    x, y = np.array(x, dtype=float), np.array(PRICE_1, dtype=float)
    w = b = 0.0
    out = [(w, b)]
    for _ in range(steps):
        r = y - (w * x + b)
        w, b = w - lr * -2 * (r * x).mean(), b - lr * -2 * r.mean()
        out.append((w, b))
    return out


def picture(labels: dict[str, str]) -> str:
    """Two contour plots of the loss over w and b: with the area in m², a long, narrow valley, and standardized, a
    round one. Each with 50 steps of gradient descent from (0, 0), at the largest learning rate that still works well."""
    import matplotlib.pyplot as plt
    import numpy as np
    from everything_ai import plots
    a, p = np.array(AREA, dtype=float), np.array(PRICE_1, dtype=float)
    z = (a - a.mean()) / a.std()
    panels = [(a, 0.00012, (-1, 7), (-100, 400), [300, 1000, 3000, 10000, 30000, 60000], labels["raw"]),
              (z, 0.3, (-60, 400), (-60, 400), [300, 1000, 3000, 10000, 30000, 60000], labels["scaled"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, (x, lr, (w0, w1), (b0, b1), levels, title) in zip(axes, panels):
        W, B = np.meshgrid(np.linspace(w0, w1, 200), np.linspace(b0, b1, 200))
        L = ((p - (W[..., None] * x + B[..., None])) ** 2).mean(-1)
        ax.contour(W, B, L, levels=levels, colors=[plots.LOSS], linewidths=1, linestyles="-")
        steps = _path(x, lr, 50)
        ax.plot([w for w, _ in steps], [b for _, b in steps], color=plots.INK_3, linewidth=1)
        ax.scatter([w for w, _ in steps], [b for _, b in steps], color=plots.PARAM, s=8, zorder=3)
        for k in (0, 50):
            w, b = steps[k]
            ax.scatter([w], [b], color=plots.PARAM, s=36, zorder=4)
            ax.text(w + 0.04 * (w1 - w0), b - 0.07 * (b1 - b0), labels["step"].format(k=k), color=plots.PARAM_TEXT, fontsize=9.5)
        best = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), p, rcond=None)[0]
        ax.scatter([best[0]], [best[1]], color=plots.LOSS, marker="x", s=40, zorder=5)
        ax.set_xlim(w0, w1)
        ax.set_ylim(b0, b1)
        ax.set_xlabel("w")
        ax.set_ylabel("b")
        ax.set_title(title)
    fig.tight_layout()
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "z-value" is the choice question in the lesson.
REQUIRED = ["z-value", "standardize", "predict-new"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_S_HEAD = "import numpy as np\n\nX = np.array([[6, 480], [10, 450], [2, 540], [8, 420]])\n"
_S_START = _S_HEAD + "Z = ...\nZ\n"
_S = _S_HEAD + "Z = (X - X.mean(axis=0)) / X.std(axis=0)\nZ"


def _std(new: str) -> str:
    return _S.replace("Z = (X - X.mean(axis=0)) / X.std(axis=0)", new)


_P_HEAD = ("import numpy as np\n\nX = np.array(" + repr(HOUSES_RAW[:4])[:-1] + ",\n              " + repr(HOUSES_RAW[4:])[1:] + ")\nmean = X.mean(axis=0)\nstd = X.std(axis=0)\n"
           "w = np.array([53.39, 8.28, -14.89])\nb = 221.25\nnew = np.array([100, 1, 500])\n")
_P_START = _P_HEAD + "pred = ...\npred\n"
_P = _P_HEAD + "pred = (new - mean) / std @ w + b\npred"


def _pred(new: str) -> str:
    return _P.replace("pred = (new - mean) / std @ w + b", new)


EXERCISES = {
    "standardize": CodeBox(
        name="standardize",
        start=_S_START,
        solution=_S,
        keep=["X = np.array([[6, 480], [10, 450], [2, 540], [8, 420]])"],
        # The check also runs the box with other students: numbers typed in by hand do not change with them.
        others=[{"X": Code("np.array([[4, 400], [7, 510], [9, 450]])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("rows", syntax=r"axis\s*=\s*1\b",
                    en="Python stops: the shapes do not fit. With `axis=1` you take the mean of each row, one per student, but each feature is a column. Use `axis=0`: one mean and one standard deviation per column.",
                    nl="Python stopt: de shapes passen niet. Met `axis=1` neem je het gemiddelde van elke rij, één per leerling, maar elke feature is een kolom. Gebruik `axis=0`: één gemiddelde en één standaardafwijking per kolom."),
            Mistake("brackets", wrong=_std("Z = X - X.mean(axis=0) / X.std(axis=0)"),
                    en="Your last line gives `{got}`. Python divides first, and subtracts after: only the mean is divided by the standard deviation. Put brackets around `X - X.mean(axis=0)`.",
                    nl="Je laatste regel geeft `{got}`. Python deelt eerst, en trekt daarna af: alleen het gemiddelde wordt gedeeld door de standaardafwijking. Zet haakjes om `X - X.mean(axis=0)`."),
            Mistake("no-axis", wrong=_std("Z = (X - X.mean()) / X.std()"),
                    en="Your last line gives `{got}`: without `axis`, NumPy takes one mean and one standard deviation over all numbers together, hours and minutes mixed. Each feature needs its own: `axis=0`.",
                    nl="Je laatste regel geeft `{got}`: zonder `axis` neemt NumPy één gemiddelde en één standaardafwijking over alle getallen samen, uren en minuten door elkaar. Elke feature heeft een eigen gemiddelde en standaardafwijking nodig: `axis=0`."),
            Mistake("no-center", wrong=_std("Z = X / X.std(axis=0)"),
                    en="Your last line gives `{got}`: every number is divided by the standard deviation, but the mean was not taken off first. Subtract `X.mean(axis=0)` before you divide.",
                    nl="Je laatste regel geeft `{got}`: elk getal is gedeeld door de standaardafwijking, maar het gemiddelde is er niet eerst afgetrokken. Trek `X.mean(axis=0)` af voordat je deelt."),
            Mistake("no-scale", wrong=_std("Z = X - X.mean(axis=0)"),
                    en="Your last line gives `{got}`: the mean is taken off, so each column is around 0, but the minutes still swing far more than the hours. Divide by the standard deviation as well.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde is eraf, dus elke kolom ligt rond 0, maar de minuten schommelen nog veel meer dan de uren. Deel ook door de standaardafwijking."),
            Mistake("ddof", wrong=_std("Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)"),
                    en="Your last line gives `{got}`: with `ddof=1`, the standard deviation divides by one less than the number of students. Standardization in this course uses `X.std(axis=0)`, which divides by the number of values: leave out `ddof=1`.",
                    nl="Je laatste regel geeft `{got}`: met `ddof=1` deelt de standaardafwijking door één minder dan het aantal leerlingen. Standaardisatie gebruikt in deze course `X.std(axis=0)`, die deelt door het aantal waarden: laat `ddof=1` weg."),
            Mistake("by-mean", wrong=_std("Z = (X - X.mean(axis=0)) / X.mean(axis=0)"),
                    en="Your last line gives `{got}`: you divide by the mean. Standardization divides by the standard deviation: `X.std(axis=0)`.",
                    nl="Je laatste regel geeft `{got}`: je deelt door het gemiddelde. Standaardisatie deelt door de standaardafwijking: `X.std(axis=0)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the standardized `X`: every column minus its mean, divided by its standard deviation.",
                      "nl": "Vervang de `...` door de gestandaardiseerde `X`: elke kolom min zijn gemiddelde, gedeeld door zijn standaardafwijking."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other students by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere leerlingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `Z` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `Z` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does every column get its own mean taken off, and is it then divided by its own standard deviation?",
                      "nl": "Je laatste regel geeft `{got}`. Gaat er van elke kolom zijn eigen gemiddelde af, en wordt hij dan gedeeld door zijn eigen standaardafwijking?"},
            "other": {"en": "Right for these students. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `X` itself?",
                      "nl": "Goed voor deze leerlingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X` zelf?"},
            "other_error": {"en": "Right for these students. But with {where}, your code stops with an error. Does it work for any number of students?",
                            "nl": "Goed voor deze leerlingen. Maar met {where} stopt je code met een fout. Werkt hij voor elk aantal leerlingen?"},
            "correct": {"en": "Both columns now have mean 0 and standard deviation 1: hours and minutes are on the same scale. The student who slept 540 minutes gets 1.5 for sleep, far above the rest. The check also tried three other students.",
                        "nl": "Beide kolommen hebben nu gemiddelde 0 en standaardafwijking 1: uren en minuten staan op dezelfde schaal. De leerling die 540 minuten sliep, krijgt 1,5 voor slaap, ver boven de rest. De check probeerde ook drie andere leerlingen."},
        },
    ),
    "predict-new": CodeBox(
        name="predict-new",
        start=_P_START,
        solution=_P,
        keep=["mean = X.mean(axis=0)", "std = X.std(axis=0)", "w = np.array([53.39, 8.28, -14.89])", "b = 221.25", "new = np.array([100, 1, 500])"],
        # The check also runs the box with another new house.
        others=[{"new": Code("np.array([70, 0, 1000])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("raw", wrong=_pred("pred = new @ w + b"),
                    en="Your last line gives `{got}`: a price far below 0. The model learned its weights on standardized features, so the new house needs the same: `(new - mean) / std` first.",
                    nl="Je laatste regel geeft `{got}`: een prijs ver onder 0. Het model leerde zijn weights op gestandaardiseerde features, dus het nieuwe huis heeft dat ook nodig: eerst `(new - mean) / std`."),
            Mistake("own-stats", wrong=_pred("pred = (new - new.mean()) / new.std() @ w + b"),
                    en="Your last line gives `{got}`: you standardized the new house with its own mean and standard deviation, which mixes m², garden and meters. Use `mean` and `std` of the training houses.",
                    nl="Je laatste regel geeft `{got}`: je standaardiseerde het nieuwe huis met zijn eigen gemiddelde en standaardafwijking, en dat haalt m², tuin en meters door elkaar. Gebruik `mean` en `std` van de trainingshuizen."),
            Mistake("brackets", wrong=_pred("pred = new - mean / std @ w + b"),
                    en="Your last line gives `{got}`. Python works out `mean / std @ w` first, before it subtracts. Put brackets around `new - mean`.",
                    nl="Je laatste regel geeft `{got}`. Python rekent eerst `mean / std @ w` uit, voordat hij aftrekt. Zet haakjes om `new - mean`."),
            Mistake("no-bias", wrong=_pred("pred = (new - mean) / std @ w"),
                    en="Your last line gives `{got}`: the bias term is missing. Add `b`.",
                    nl="Je laatste regel geeft `{got}`: de bias term ontbreekt. Tel `b` erbij op."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the predicted price of `new`: standardize it with `mean` and `std`, then take `@ w` and add `b`.",
                      "nl": "Vervang de `...` door de voorspelde prijs van `new`: standaardiseer hem met `mean` en `std`, doe dan `@ w` en tel `b` erbij op."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another new house by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een ander nieuw huis."},
            "no_value": {"en": "The last line of the box shows nothing. Put `pred` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `pred` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you standardize `new` with `mean` and `std`, then take `@ w`, and add `b`?",
                      "nl": "Je laatste regel geeft `{got}`. Standaardiseer je `new` met `mean` en `std`, doe je dan `@ w`, en tel je `b` erbij op?"},
            "other": {"en": "Right for this house. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `new` itself?",
                      "nl": "Goed voor dit huis. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `new` zelf?"},
            "other_error": {"en": "Right for this house. But with {where}, your code stops with an error. Do you work it out from `new` itself?",
                            "nl": "Goed voor dit huis. Maar met {where} stopt je code met een fout. Reken je het uit met `new` zelf?"},
            "correct": {"en": "About 278.6 thousand euros for 100 m², a garden and 500 m to the station. The check also tried a house of 70 m² without a garden, 1000 m from the station: about 181.4.",
                        "nl": "Ongeveer 278,6 duizend euro voor 100 m², een tuin en 500 m tot het station. De check probeerde ook een huis van 70 m² zonder tuin, op 1000 m van het station: ongeveer 181,4."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("standardize", _S_START, "empty", None),
    ("standardize", _S, "correct", None),
    ("standardize", _std("mean = X.mean(axis=0)\nstd = X.std(axis=0)\nZ = (X - mean) / std"), "correct", None),
    ("standardize", _std("Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)"), "correct", None),
    ("standardize", _std("Z = (X - X.mean(axis=1)) / X.std(axis=1)"), "error", "rows"),
    ("standardize", _std("Z = X - X.mean(axis=0) / X.std(axis=0)"), "wrong", "brackets"),
    ("standardize", _std("Z = (X - X.mean()) / X.std()"), "wrong", "no-axis"),
    ("standardize", _std("Z = X / X.std(axis=0)"), "wrong", "no-center"),
    ("standardize", _std("Z = X - X.mean(axis=0)"), "wrong", "no-scale"),
    ("standardize", _std("Z = (X - X.mean(axis=0)) / X.mean(axis=0)"), "wrong", "by-mean"),
    ("standardize", _std("Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)"), "wrong", "ddof"),
    ("standardize", _S.replace("[2, 540]", "[3, 540]"), "wrong", "changed-line"),
    ("predict-new", _P.replace("[100, 1, 500]", "[70, 0, 1000]"), "wrong", "changed-line"),
    ("standardize", _S_HEAD + "Z = (X - X.mean(axis=0)) / X.std(axis=0)\nprint(Z)", "wrong", "print-last"),
    ("predict-new", _P_START, "empty", None),
    ("predict-new", _P, "correct", None),
    ("predict-new", _pred("z = (new - mean) / std\npred = z @ w + b"), "correct", None),
    ("predict-new", _pred("pred = np.dot((new - mean) / std, w) + b"), "correct", None),
    ("predict-new", _pred("pred = new @ w + b"), "wrong", "raw"),
    ("predict-new", _pred("pred = (new - new.mean()) / new.std() @ w + b"), "wrong", "own-stats"),
    ("predict-new", _pred("pred = new - mean / std @ w + b"), "wrong", "brackets"),
    ("predict-new", _pred("pred = (new - mean) / std @ w"), "wrong", "no-bias"),
    ("predict-new", _pred("pred = 278.5566699865792"), "almost", "other-value"),
]
