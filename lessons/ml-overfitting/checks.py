# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-overfitting: the test MSE of a curve of a given degree, fitted on the training set. And the
picture of the temperatures of the worked example, with a curve of the degree on the slider and the training and
test MSE of every degree."""
import re

from everything_ai.check import Code, CodeBox, Mistake

SEED, N, N_TRAIN, DEGREES = 75, 32, 12, range(1, 10)   # as in the worked example


def _data():
    """The 32 made-up temperatures of the worked example, split as it splits them."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    hours = 24 * rng.random(N)
    temps = 20 - 0.05 * (hours - 15) ** 2 + 3 * (rng.random(N) - 0.5)
    order = rng.permutation(N)
    return hours[order[:N_TRAIN]], temps[order[:N_TRAIN]], hours[order[N_TRAIN:]], temps[order[N_TRAIN:]]


X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = _data()


def _fit(degree: int):
    import numpy as np
    return np.polyfit(X_TRAIN, Y_TRAIN, degree)


def errors(degree: int) -> tuple[float, float]:
    """The MSE of the curve of this degree on the training set and on the test set."""
    import numpy as np
    w = _fit(degree)
    return (float(np.mean((Y_TRAIN - np.polyval(w, X_TRAIN)) ** 2)), float(np.mean((Y_TEST - np.polyval(w, X_TEST)) ** 2)))


ERRORS = {d: errors(d) for d in DEGREES}


def picture(degree: int, labels: dict[str, str]) -> str:
    """The temperatures, training measurements filled and test measurements open, with the curve of this degree; and
    the training MSE (filled dots) and the test MSE (open dots) of every degree, the chosen one larger. Drawn as SVG text,
    for every stop of the slider once (L.explore); dragging along the MSE chart chooses a degree."""
    import numpy as np
    from everything_ai import chart
    w = _fit(degree)
    data = chart.Panel((0, 24), (6, 24), title=labels["data"], xlabel=labels["hour"], ylabel=labels["temp"], xticks=[0, 6, 12, 18, 24])
    data.points(X_TRAIN, Y_TRAIN, "data", 4.5)
    data.points(X_TEST, Y_TEST, "data", 4.5, hollow=True)
    xs = [24 * i / 240 for i in range(241)]
    data.line(xs, list(np.polyval(w, xs)), "pred", 2.4)
    mse = chart.Panel((0.5, 9.5), (0, 24), title=labels["mse"], xlabel=labels["degree"], ylabel=labels["loss"], xticks=list(DEGREES))
    ds = list(DEGREES)
    train = [ERRORS[d][0] for d in ds]
    test = [ERRORS[d][1] for d in ds]
    mse.line(ds, train, "loss", 2.0)
    mse.line(ds, test, "loss", 2.0, dashed=True)
    mse.points(ds, train, "loss", 3.5)
    mse.points(ds, test, "loss", 3.5, hollow=True)
    mse.points([degree], [ERRORS[degree][0]], "loss", 6.5)
    mse.points([degree], [ERRORS[degree][1]], "loss", 6.5, hollow=True)
    return chart.figure([data, mse], alt=labels["alt"], width=330, height=290, drag=1)   # drag along the MSE; two fit a play block side by side


# The lesson is done when these are correct. "which-overfits" is the choice question in the lesson.
REQUIRED = ["which-overfits", "test-mse"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_T_HEAD = ("import numpy as np\n\nx_train = np.array([8.4, 23.9, 6.0, 22.7, 5.1, 20.1, 22.4, 4.2, 9.9, 0.5, 20.8, 23.6])\n"
           "y_train = np.array([17.8, 15.2, 15.5, 15.5, 14.1, 19.4, 18.7, 15.5, 18.6, 8.8, 18.4, 17.3])\n"
           "x_test = np.array([10.3, 4.9, 16.0, 2.2, 16.4, 21.8])\ny_test = np.array([19.3, 16.2, 20.7, 12.4, 19.7, 16.3])\n"
           "degree = 2\n")
_T_START = _T_HEAD + "w = ...\nmse_test = ...\nmse_test\n"
_T_BODY = "w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean((y_test - np.polyval(w, x_test)) ** 2)\n"
_T = _T_HEAD + _T_BODY + "mse_test"


def _t(body: str) -> str:
    return _T_HEAD + body + "mse_test"


def _no_square() -> float:
    """The mean residual on the test set, without squaring, for degree 2: the outcome of the mistake "no-square"."""
    import numpy as np
    names = {"np": np}
    exec(_T_HEAD + "w = np.polyfit(x_train, y_train, degree)\n", names)
    return float(np.mean(names["y_test"] - np.polyval(names["w"], names["x_test"])))


_NO_SQUARE = _no_square()


def _same(got, want: float) -> bool:
    try:
        return abs(float(got) - want) < 1e-9
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "test-mse": CodeBox(
        name="test-mse",
        start=_T_START,
        solution=_T,
        # The check also runs the box with another degree: a number typed in by hand does not change with it.
        others=[{"degree": Code("4")}],
        # Another degree is for the scratchpad: the check needs `degree = 2` to know the right outcome.
        keep=["degree = 2"],
        first=[_DOTS],
        mistakes=[
            Mistake("on-train", wrong=_t("w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean((y_train - np.polyval(w, x_train)) ** 2)\n"),
                    en="Your last line gives `{got}`: the MSE on the training measurements, which the curve learned from. Measure on the test set: `x_test` and `y_test`.",
                    nl="Je laatste regel geeft `{got}`: de MSE op de trainingsmetingen, waarvan de curve leerde. Meet op de test set: `x_test` en `y_test`."),
            Mistake("fit-on-test", wrong=_t("w = np.polyfit(x_test, y_test, degree)\nmse_test = np.mean((y_test - np.polyval(w, x_test)) ** 2)\n"),
                    en="Your last line gives `{got}`: you fit the curve on the test set itself, so it has seen the measurements it is tested on. Fit on `x_train` and `y_train`.",
                    nl="Je laatste regel geeft `{got}`: je laat de curve leren van de test set zelf, dus hij kent de metingen waarop hij getest wordt. Laat hem leren van `x_train` en `y_train`."),
            Mistake("no-square", wrong=_t("w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean(y_test - np.polyval(w, x_test))\n"),
                    matches=lambda got, text: _same(got, -_NO_SQUARE),
                    en="Your last line gives `{got}`: the mean residual, without squaring. Positive and negative residuals cancel. Square each residual first: `** 2`.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde residu, zonder kwadraat. Positieve en negatieve residuen vallen tegen elkaar weg. Kwadrateer eerst elk residu: `** 2`."),
            Mistake("swapped-polyval", syntax=r"np\.polyval\(\s*x_\w+\s*,",
                    en="Python stops: `np.polyval` takes the weights first and the hours second: `np.polyval(w, x_test)`.",
                    nl="Python stopt: `np.polyval` krijgt eerst de weights en dan de uren: `np.polyval(w, x_test)`."),
        ],
        texts={
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` as it was: put it back. The check tries degree 4 itself; try other degrees in the scratchpad.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. Graad 4 probeert de check zelf; andere graden probeer je in het kladblok."},
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the weights of a curve of degree `degree` fitted on the training set, and its MSE on the test set.",
                      "nl": "Vervang de twee `...`: de weights van een curve van graad `degree`, geleerd van de training set, en zijn MSE op de test set."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `mse_test` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `mse_test` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you fit with `np.polyfit` on the training set, and take the mean of the squared residuals on the test set?",
                      "nl": "Je laatste regel geeft `{got}`. Laat je `np.polyfit` leren van de training set, en neem je het gemiddelde van de gekwadrateerde residuen op de test set?"},
            "other": {"en": "Right for degree 2. But with {where}, your last line gives `{got}`, and it should give about `1.01`. Do you use `degree` itself in `np.polyfit`?",
                      "nl": "Goed voor graad 2. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `1.01` zijn. Gebruik je `degree` zelf in `np.polyfit`?"},
            "other_error": {"en": "Right for degree 2. But with {where}, your code stops with an error. Do you use `degree` itself in `np.polyfit`?",
                            "nl": "Goed voor graad 2. Maar met {where} stopt je code met een fout. Gebruik je `degree` zelf in `np.polyfit`?"},
            "correct": {"en": "The test MSE is about 0.96. The check also tried degree 4: about 1.01. Copy your code into the scratchpad and try `degree = 9`: about 37.",
                        "nl": "De test-MSE is ongeveer 0,96. De check probeerde ook graad 4: ongeveer 1,01. Kopieer je code naar het kladblok en probeer `degree = 9`: ongeveer 37."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("test-mse", _T_START, "empty", None),
    ("test-mse", _T, "correct", None),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, degree)\npred = np.polyval(w, x_test)\nmse_test = ((pred - y_test) ** 2).mean()\n"), "correct", None),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, deg=degree)\nmse_test = np.sum((y_test - np.polyval(w, x_test)) ** 2) / len(y_test)\n"), "correct", None),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean((y_train - np.polyval(w, x_train)) ** 2)\n"), "wrong", "on-train"),
    ("test-mse", _t("w = np.polyfit(x_test, y_test, degree)\nmse_test = np.mean((y_test - np.polyval(w, x_test)) ** 2)\n"), "wrong", "fit-on-test"),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean(y_test - np.polyval(w, x_test))\n"), "wrong", "no-square"),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean(np.polyval(w, x_test) - y_test)\n"), "wrong", "no-square"),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, degree)\nmse_test = np.mean((y_test - np.polyval(x_test, w)) ** 2)\n"), "error", "swapped-polyval"),
    ("test-mse", _t("w = np.polyfit(x_train, y_train, 2)\nmse_test = np.mean((y_test - np.polyval(w, x_test)) ** 2)\n"), "almost", "other-value"),
    ("test-mse", _T.replace("degree = 2", "degree = 9"), "wrong", "changed-line"),
]
