# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-baselines: the accuracy of always predicting the most common class of the training set, and
the mean squared error of always predicting the mean training label. And the picture of the houses of the worked
example, with a line and the baseline."""
import re

from everything_ai.check import Code, CodeBox, Mistake


def _houses():
    """The forty houses of ml-train-test-split, split the same way."""
    import numpy as np
    rng = np.random.default_rng(18)
    sizes = 0.5 + 1.5 * rng.random(size=40)
    prices = 60 + 150 * sizes + 80 * (rng.random(size=40) - 0.5)
    order = rng.permutation(40)
    return sizes[order[:32]], prices[order[:32]], sizes[order[32:]], prices[order[32:]]


X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = _houses()


def picture(labels: dict[str, str]) -> str:
    """The 8 test houses (open) and the 32 training houses (filled), the line learned from the training houses, and the
    baseline: the mean training price for every house, a flat dashed line."""
    from everything_ai import chart
    w = ((X_TRAIN - X_TRAIN.mean()) * (Y_TRAIN - Y_TRAIN.mean())).mean() / ((X_TRAIN - X_TRAIN.mean()) ** 2).mean()
    b = Y_TRAIN.mean() - w * X_TRAIN.mean()
    c = chart.Panel((0.4, 2.1), (100, 400), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    c.line([0.45, 2.05], [w * 0.45 + b, w * 2.05 + b], "pred", 2.4)
    c.line([0.45, 2.05], [Y_TRAIN.mean(), Y_TRAIN.mean()], "pred", 2.0, dashed=True)
    c.points(X_TRAIN, Y_TRAIN, "data", 4)
    c.points(X_TEST, Y_TEST, "data", 4.5, hollow=True)
    c.text(1.85, Y_TRAIN.mean() - 18, labels["baseline"], "ink-2")
    return chart.figure([c], alt=labels["alt"], width=560, height=320)


# The lesson is done when these are correct. "rare-disease" is the choice question in the lesson.
REQUIRED = ["rare-disease", "majority", "mean-baseline"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# Counting the classes of the test set: the outcome can even be right, as here, so the check looks at the code first.
_FROM_TEST = Mistake("from-test", matches=lambda got, text: re.search(r"^[^#\n]*(?:\[\s*y_test\s*\]|bincount\(\s*y_test)", text, re.MULTILINE) is not None,
                     en="You use the test set to find the most common class. The baseline may only learn from the training set, like any model: take the class with the largest count in `counts`.",
                     nl="Je gebruikt de test set om de meest voorkomende klasse te vinden. De baseline mag alleen van de training set leren, net als elk model: neem de klasse met de grootste telling in `counts`.")

_C_HEAD = ("import numpy as np\n\ny_train = np.array([0, 1, 1, 2, 1, 0, 1, 1, 2, 1, 1, 0])\ny_test = np.array([1, 1, 0, 2, 1, 1, 1, 0])\n"
           "counts = np.eye(3)[y_train].sum(axis=0)\n")
_C_START = _C_HEAD + "most_common = ...\naccuracy = ...\naccuracy\n"
_C_BODY = "most_common = counts.argmax()\naccuracy = np.mean(y_test == most_common)\n"
_C = _C_HEAD + _C_BODY + "accuracy"


def _c(body: str) -> str:
    return _C_HEAD + body + "accuracy"


_M_HEAD = "import numpy as np\n\ny_train = np.array([210, 250, 180, 300, 260, 240])\ny_test = np.array([220, 280, 190])\n"
_M_START = _M_HEAD + "baseline = ...\nmse = ...\nmse\n"
_M_BODY = "baseline = y_train.mean()\nmse = np.mean((y_test - baseline) ** 2)\n"
_M = _M_HEAD + _M_BODY + "mse"


def _m(body: str) -> str:
    return _M_HEAD + body + "mse"


EXERCISES = {
    "majority": CodeBox(
        name="majority",
        start=_C_START,
        solution=_C,
        keep=["counts = np.eye(3)[y_train].sum(axis=0)"],
        # The check also runs the box with other training labels: a class typed in by hand does not change with them.
        others=[{"y_train": Code("np.array([2, 2, 0, 2, 1, 0, 2, 2, 1, 2, 0, 2])")}],
        first=[_DOTS, _FROM_TEST],
        mistakes=[
            Mistake("count-itself", wrong=_c("most_common = counts.max()\naccuracy = np.mean(y_test == most_common)\n"),
                    en="Your last line gives `{got}`: not one test label equals your `most_common`, so it is not one of the classes 0, 1 and 2. `counts.max()` is how often the most common class occurs, 7; `counts.argmax()` gives its place: the class.",
                    nl="Je laatste regel geeft `{got}`: geen enkel testlabel is gelijk aan je `most_common`, dus dat is geen van de klassen 0, 1 en 2. `counts.max()` is hoe vaak de meest voorkomende klasse voorkomt, 7; `counts.argmax()` geeft zijn plek: de klasse."),
            Mistake("rarest", wrong=_c("most_common = counts.argmin()\naccuracy = np.mean(y_test == most_common)\n"),
                    en="Your last line gives `{got}`: the accuracy of always predicting class 2, the rarest class in the training set. The baseline always predicts the most common one: the place of the largest count.",
                    nl="Je laatste regel geeft `{got}`: de accuracy als je altijd klasse 2 voorspelt, de zeldzaamste klasse in de training set. De baseline voorspelt altijd de meest voorkomende: de plek van de grootste telling."),
            Mistake("on-train", wrong=_c("most_common = counts.argmax()\naccuracy = np.mean(y_train == most_common)\n"),
                    en="Your last line gives `{got}`: the accuracy on the training set. Compare the baseline with the model on the same test set: `y_test`.",
                    nl="Je laatste regel geeft `{got}`: de accuracy op de training set. Vergelijk de baseline met het model op dezelfde test set: `y_test`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the most common class of the training set, and the accuracy on the test set of always predicting it.",
                      "nl": "Vervang de twee `...`: de meest voorkomende klasse van de training set, en de accuracy op de test set als je die altijd voorspelt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other training labels by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere trainingslabels."},
            "no_value": {"en": "The last line of the box shows nothing. Put `accuracy` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `accuracy` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `most_common` the place of the largest count, and `accuracy` the share of test labels equal to it?",
                      "nl": "Je laatste regel geeft `{got}`. Is `most_common` de plek van de grootste telling, en `accuracy` het deel van de testlabels dat eraan gelijk is?"},
            "other": {"en": "Right for these labels. But with {where}, your last line gives `{got}`, and it should give `0.125`. Do you take the most common class from `counts` itself?",
                      "nl": "Goed voor deze labels. Maar met {where} geeft je laatste regel `{got}`, en dat moet `0.125` zijn. Neem je de meest voorkomende klasse uit `counts` zelf?"},
            "other_error": {"en": "Right for these labels. But with {where}, your code stops with an error. Do you take the most common class from `counts` itself?",
                            "nl": "Goed voor deze labels. Maar met {where} stopt je code met een fout. Neem je de meest voorkomende klasse uit `counts` zelf?"},
            "correct": {"en": "Class 1 is the most common in training, 7 of 12. Always predicting it gets 5 of the 8 test examples right: 0.625. A model has to beat that. The check also tried other training labels.",
                        "nl": "Klasse 1 komt in de training set het meest voor, 7 van de 12. Altijd die voorspellen geeft 5 van de 8 testvoorbeelden goed: 0,625. Een model moet dat verslaan. De check probeerde ook andere trainingslabels."},
        },
    ),
    "mean-baseline": CodeBox(
        name="mean-baseline",
        start=_M_START,
        solution=_M,
        # The check also runs the box with other training prices: a number typed in by hand does not change with them.
        others=[{"y_train": Code("np.array([150, 200, 250, 300, 350, 400])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("test-mean", wrong=_m("baseline = y_test.mean()\nmse = np.mean((y_test - baseline) ** 2)\n"),
                    en="Your last line gives `{got}`: you take the mean of the test prices. The baseline may only learn from the training set: `y_train.mean()`.",
                    nl="Je laatste regel geeft `{got}`: je neemt het gemiddelde van de testprijzen. De baseline mag alleen van de training set leren: `y_train.mean()`."),
            Mistake("no-square", wrong=_m("baseline = y_train.mean()\nmse = np.mean(y_test - baseline)\n"),
                    en="Your last line gives `{got}`: the mean residual, without squaring. Square each residual first: `** 2`.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde residu, zonder kwadraat. Kwadrateer eerst elk residu: `** 2`."),
            Mistake("on-train", wrong=_m("baseline = y_train.mean()\nmse = np.mean((y_train - baseline) ** 2)\n"),
                    en="Your last line gives `{got}`: the MSE on the training set. Measure the baseline on the test set, like the model: `y_test`.",
                    nl="Je laatste regel geeft `{got}`: de MSE op de training set. Meet de baseline op de test set, net als het model: `y_test`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the mean training price, and the MSE on the test set of predicting it for every house.",
                      "nl": "Vervang de twee `...`: de gemiddelde trainingsprijs, en de MSE op de test set als je die voor elk huis voorspelt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `mse` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `mse` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `baseline` the mean of `y_train`, and `mse` the mean of the squared differences between `y_test` and it?",
                      "nl": "Je laatste regel geeft `{got}`. Is `baseline` het gemiddelde van `y_train`, en `mse` het gemiddelde van de gekwadrateerde verschillen tussen `y_test` en dat getal?"},
            "other": {"en": "Right for these prices. But with {where}, your last line gives `{got}`, and it should give `3425`. Do you work out the mean from `y_train` itself?",
                      "nl": "Goed voor deze prijzen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `3425` zijn. Reken je het gemiddelde uit met `y_train` zelf?"},
            "other_error": {"en": "Right for these prices. But with {where}, your code stops with an error. Do you work out the mean from `y_train` itself?",
                            "nl": "Goed voor deze prijzen. Maar met {where} stopt je code met een fout. Reken je het gemiddelde uit met `y_train` zelf?"},
            "correct": {"en": "The mean training price is 240, and predicting 240 for every test house gives an MSE of 1500. Any model for these prices has to get below that. The check also tried other training prices.",
                        "nl": "De gemiddelde trainingsprijs is 240, en voor elk testhuis 240 voorspellen geeft een MSE van 1500. Elk model voor deze prijzen moet daaronder komen. De check probeerde ook andere trainingsprijzen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("majority", _C_START, "empty", None),
    ("majority", _C, "correct", None),
    ("majority", _c("most_common = np.argmax(counts)\naccuracy = (y_test == most_common).mean()\n"), "correct", None),
    ("majority", _c("most_common = counts.argmax()\naccuracy = np.sum(y_test == most_common) / len(y_test)\n"), "correct", None),
    ("majority", _c("most_common = np.eye(3)[y_test].sum(axis=0).argmax()\naccuracy = np.mean(y_test == most_common)\n"), "wrong", "from-test"),
    ("majority", _c("# not np.eye(3)[y_test]: that would use the test set\nmost_common = counts.argmax()\naccuracy = np.mean(y_test == most_common)\n"), "correct", None),
    ("majority", _c("most_common = y_train.max()\naccuracy = np.mean(y_test == most_common)\n"), "wrong", "rarest"),
    ("majority", _c("most_common = counts.max()\naccuracy = np.mean(y_test == most_common)\n"), "wrong", "count-itself"),
    ("majority", _c("most_common = counts.argmin()\naccuracy = np.mean(y_test == most_common)\n"), "wrong", "rarest"),
    ("majority", _c("most_common = counts.argmax()\naccuracy = np.mean(y_train == most_common)\n"), "wrong", "on-train"),
    ("majority", _c("most_common = 1\naccuracy = np.mean(y_test == most_common)\n"), "almost", "other-value"),
    ("mean-baseline", _M_START, "empty", None),
    ("mean-baseline", _M, "correct", None),
    ("mean-baseline", _m("baseline = np.mean(y_train)\nmse = ((y_test - baseline) ** 2).mean()\n"), "correct", None),
    ("mean-baseline", _m("baseline = y_test.mean()\nmse = np.mean((y_test - baseline) ** 2)\n"), "wrong", "test-mean"),
    ("mean-baseline", _m("baseline = y_train.mean()\nmse = np.mean(y_test - baseline)\n"), "wrong", "no-square"),
    ("mean-baseline", _m("baseline = y_train.mean()\nmse = np.mean((y_train - baseline) ** 2)\n"), "wrong", "on-train"),
    ("mean-baseline", _m("baseline = 240\nmse = np.mean((y_test - baseline) ** 2)\n"), "almost", "other-value"),
]
