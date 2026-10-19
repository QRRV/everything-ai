# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-sklearn: fit a LinearRegression on students, and compare it with a model that always
predicts the mean, with the same two steps."""
import re

from everything_ai.check import Code, CodeBox, Mistake

# The houses of the lesson on linear regression, as a table: area in hundreds of m², garden (1 = yes), distance to
# the station in km, and the price in thousands of euros.
HOUSES = {"area": [0.62, 0.85, 1.10, 0.48, 0.95, 1.30, 0.73, 0.58], "garden": [0, 1, 1, 0, 0, 1, 1, 0],
          "distance": [0.4, 1.2, 0.8, 0.3, 1.5, 0.6, 0.9, 1.1], "price": [191, 218, 290, 162, 215, 333, 211, 150]}

# The lesson is done when these are correct. "predict-shape" is the choice question in the lesson.
REQUIRED = ["predict-shape", "fit-students", "mean-model"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_DF = 'df = pd.DataFrame({"hours": [6, 10, 2, 8, 4, 7], "sleep": [8, 7, 9, 6, 7, 8], "grade": [6.6, 7.8, 4.8, 6.6, 5.4, 6.8]})\n'
_OTHER_DF = Code('pd.DataFrame({"hours": [6, 10, 2, 8, 4, 7], "sleep": [8, 7, 9, 6, 7, 8], "grade": [5.5, 8.0, 4.0, 7.5, 5.0, 6.5]})')
_IMPORTS = "import pandas as pd\nfrom sklearn.linear_model import LinearRegression\n"

_F_HEAD = _IMPORTS + "\n" + _DF + 'y = df["grade"]\n'
_F_START = _F_HEAD + "X = ...\n...\nmodel.coef_\n"
_F = _F_HEAD + 'X = df[["hours", "sleep"]]\nmodel = LinearRegression()\nmodel.fit(X, y)\nmodel.coef_'


def _f(new: str) -> str:
    return _F_HEAD + new + "\nmodel.coef_"


def _length(got, n: int) -> bool:
    try:
        return len(got) == n
    except TypeError:
        return False


_M_HEAD = (_IMPORTS + "from sklearn.dummy import DummyRegressor\n\n" + _DF + 'X = df[["hours", "sleep"]]\ny = df["grade"]\n\n'
           "model = LinearRegression()\nmodel.fit(X, y)\nmse_model = ((y - model.predict(X)) ** 2).mean()\n\n")
_M_START = _M_HEAD + "mean_model = DummyRegressor()\n...\n[mse_model, mse_mean]\n"
_M = _M_HEAD + "mean_model = DummyRegressor()\nmean_model.fit(X, y)\nmse_mean = ((y - mean_model.predict(X)) ** 2).mean()\n[mse_model, mse_mean]"


def _m(new: str) -> str:
    return _M_HEAD + "mean_model = DummyRegressor()\n" + new + "\n[mse_model, mse_mean]"


_NOT_FITTED = lambda e: "not fitted" in str(e) or (isinstance(e, AttributeError) and "coef_" in str(e))   # noqa: E731

EXERCISES = {
    "fit-students": CodeBox(
        name="fit-students",
        start=_F_START,
        solution=_F,
        keep=['y = df["grade"]', "model.coef_"],
        # The check also runs the box with other grades: numbers typed in by hand do not change with them.
        others=[{"df": _OTHER_DF}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-parens", syntax=r"=\s*LinearRegression\s*$",
                    en="Python stops: without the brackets, `model` is not a new model yet. `LinearRegression()`, with `()`, makes one: step 2 of the recipe.",
                    nl="Python stopt: zonder de haakjes is `model` nog geen nieuw model. `LinearRegression()`, met `()`, maakt er een: stap 2 van het recept."),
            Mistake("swapped", syntax=r"\.fit\(\s*y\s*,\s*X\s*\)",
                    en="Python stops: `fit` wants the features first and the labels second: `model.fit(X, y)`.",
                    nl="Python stopt: `fit` wil eerst de features en dan de labels: `model.fit(X, y)`."),
            Mistake("one-bracket", error=lambda e: isinstance(e, ValueError) and ("2-dimensional" in str(e) or "2D array" in str(e)),
                    en="Python stops: `fit` wants the features as a table, with one column per feature. One pair of square brackets gives a single column of numbers; two pairs give a table: `df[[\"hours\", \"sleep\"]]`.",
                    nl="Python stopt: `fit` wil de features als tabel, met één kolom per feature. Eén paar vierkante haken geeft één losse kolom getallen; twee paar geven een tabel: `df[[\"hours\", \"sleep\"]]`."),
            Mistake("two-names", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], tuple),
                    en="Python stops: `df[\"hours\", \"sleep\"]` looks for one column with both names in it. Put the names in a list, inside the brackets: `df[[\"hours\", \"sleep\"]]`.",
                    nl="Python stopt: `df[\"hours\", \"sleep\"]` zoekt één kolom met beide namen erin. Zet de namen in een lijst, binnen de haken: `df[[\"hours\", \"sleep\"]]`."),
            Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                    en="Python stops: `df` has no column with that name. Check the spelling, small letters included: the columns are `hours`, `sleep` and `grade`.",
                    nl="Python stopt: `df` heeft geen kolom met die naam. Kijk de spelling na, ook de kleine letters: de kolommen heten `hours`, `sleep` en `grade`."),
            Mistake("no-fit", error=_NOT_FITTED,
                    en="Python stops: the model has not learned anything yet, so it has no weights. Call `model.fit(X, y)` before you ask for `model.coef_`.",
                    nl="Python stopt: het model heeft nog niets geleerd, dus het heeft geen weights. Roep `model.fit(X, y)` aan voordat je om `model.coef_` vraagt."),
            Mistake("with-label", matches=lambda got, text: _length(got, 3),
                    detail=lambda got, text: {"weights": str([round(float(v), 2) + 0.0 for v in got])},
                    en="Your last line gives three weights, about `{weights}`: the model just copies the grade, the last column. The grade is the label, what the model has to predict: it cannot also be a feature. Leave `grade` out of `X`.",
                    nl="Je laatste regel geeft drie weights, ongeveer `{weights}`: het model neemt gewoon het cijfer over, de laatste kolom. Het cijfer is het label, wat het model moet voorspellen: dat kan niet ook een feature zijn. Laat `grade` weg uit `X`."),
            Mistake("order", wrong=_f('X = df[["sleep", "hours"]]\nmodel = LinearRegression()\nmodel.fit(X, y)'),
                    en="Your last line gives `{got}`: the right weights, but the other way round. `coef_` follows the order of the columns of `X`: `hours` first, then `sleep`.",
                    nl="Je laatste regel geeft `{got}`: de goede weights, maar andersom. `coef_` volgt de volgorde van de kolommen van `X`: eerst `hours`, dan `sleep`."),
            Mistake("one-feature", matches=lambda got, text: _length(got, 1),
                    en="Your last line gives `{got}`: one weight. The model should use both features: hours of study and hours of sleep.",
                    nl="Je laatste regel geeft `{got}`: één weight. Het model moet beide features gebruiken: uren leren en uren slaap."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the first `...` with the table of features, and the second with the two steps: make the model, and let it learn from `X` and `y`.",
                      "nl": "Vervang de eerste `...` door de tabel met features, en de tweede door de twee stappen: maak het model, en laat het leren van `X` en `y`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other grades by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere cijfers."},
            "no_value": {"en": "The last line of the box shows nothing. Put `model.coef_` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `model.coef_` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `X` the table with the columns `hours` and `sleep`, and did the model learn from `X` and `y` with `fit`?",
                      "nl": "Je laatste regel geeft `{got}`. Is `X` de tabel met de kolommen `hours` en `sleep`, en leerde het model van `X` en `y` met `fit`?"},
            "other": {"en": "Right for these grades. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does the model learn from `df` itself?",
                      "nl": "Goed voor deze cijfers. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Leert het model van `df` zelf?"},
            "other_error": {"en": "Right for these grades. But with {where}, your code stops with an error. Does the model learn from `df` itself?",
                            "nl": "Goed voor deze cijfers. Maar met {where} stopt je code met een fout. Leert het model van `df` zelf?"},
            "correct": {"en": "The weights are about 0.42 per hour of study and 0.26 per hour of sleep. Three lines, and no learning rate or steps: scikit-learn finds the lowest point of the loss directly. The check also tried other grades.",
                        "nl": "De weights zijn ongeveer 0,42 per uur leren en 0,26 per uur slaap. Drie regels, en geen learning rate of stappen: scikit-learn vindt het laagste punt van de loss rechtstreeks. De check probeerde ook andere cijfers."},
        },
    ),
    "mean-model": CodeBox(
        name="mean-model",
        start=_M_START,
        solution=_M,
        keep=["mse_model = ((y - model.predict(X)) ** 2).mean()", "mean_model = DummyRegressor()", "[mse_model, mse_mean]"],
        others=[{"df": _OTHER_DF}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-fit", error=_NOT_FITTED,
                    en="Python stops: `mean_model` has not learned anything yet, so it does not know the mean grade. Call `mean_model.fit(X, y)` first, just as for `model`.",
                    nl="Python stopt: `mean_model` heeft nog niets geleerd, dus hij kent het gemiddelde cijfer niet. Roep eerst `mean_model.fit(X, y)` aan, net als bij `model`."),
            Mistake("same-model", wrong=_m("mean_model.fit(X, y)\nmse_mean = ((y - model.predict(X)) ** 2).mean()"),
                    en="Your last line gives `{got}`: twice the same number. Is `mse_mean` the MSE of the predictions of `mean_model` itself?",
                    nl="Je laatste regel geeft `{got}`: twee keer hetzelfde getal. Is `mse_mean` de MSE van de voorspellingen van `mean_model` zelf?"),
            Mistake("no-mean", wrong=_m("mean_model.fit(X, y)\nmse_mean = (y - mean_model.predict(X)) ** 2"),
                    en="Your last line gives a whole column for `mse_mean`: one squared residual per student. The MSE is their mean: add `.mean()`.",
                    nl="Je laatste regel geeft een hele kolom voor `mse_mean`: één gekwadrateerd residu per leerling. De MSE is hun gemiddelde: zet er `.mean()` achter."),
            Mistake("no-square", wrong=_m("mean_model.fit(X, y)\nmse_mean = (y - mean_model.predict(X)).mean()"),
                    en="Your last line gives `{got}`: the second number is about 0, because the residuals of the mean cancel out. The MSE squares each residual first: `** 2`.",
                    nl="Je laatste regel geeft `{got}`: het tweede getal is ongeveer 0, omdat de residuen van het gemiddelde tegen elkaar wegvallen. De MSE kwadrateert eerst elk residu: `** 2`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the same steps as for `model`: let `mean_model` learn from `X` and `y`, and put its MSE in `mse_mean`.",
                      "nl": "Vervang de `...` door dezelfde stappen als bij `model`: laat `mean_model` leren van `X` en `y`, en zet zijn MSE in `mse_mean`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other grades by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere cijfers."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[mse_model, mse_mean]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[mse_model, mse_mean]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `mean_model` learn with `fit`, and is `mse_mean` the mean of the squared residuals of its predictions?",
                      "nl": "Je laatste regel geeft `{got}`. Leert `mean_model` met `fit`, en is `mse_mean` het gemiddelde van de gekwadrateerde residuen van zijn voorspellingen?"},
            "other": {"en": "Right for these grades. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `X` and `y` themselves?",
                      "nl": "Goed voor deze cijfers. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X` en `y` zelf?"},
            "other_error": {"en": "Right for these grades. But with {where}, your code stops with an error. Do you work it out from `X` and `y` themselves?",
                            "nl": "Goed voor deze cijfers. Maar met {where} stopt je code met een fout. Reken je het uit met `X` en `y` zelf?"},
            "correct": {"en": "The linear model has an MSE of about 0.016, the mean model about 0.96: always guessing 6.33 is about 60 times worse. The same two steps, fit and predict, for a different model. The check also tried other grades.",
                        "nl": "Het lineaire model heeft een MSE van ongeveer 0,016, het gemiddelde-model ongeveer 0,96: altijd 6,33 gokken is ongeveer 60 keer slechter. Dezelfde twee stappen, fit en predict, voor een ander model. De check probeerde ook andere cijfers."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fit-students", _f('X = df[["hours", "sleep"]]\nmodel = LinearRegression\nmodel.fit(X, y)'), "error", "no-parens"),
    ("fit-students", _f('X = df[["hours", "sleep"]]\nmodel = LinearRegression()\nmodel.fit(y, X)'), "error", "swapped"),
    ("fit-students", _f('X = df["hours"].values\nmodel = LinearRegression()\nmodel.fit(X, y)'), "error", "one-bracket"),
    ("fit-students", _f('X = df[["hours", "slep"]]\nmodel = LinearRegression()\nmodel.fit(X, y)'), "error", "no-column"),
    ("fit-students", _f('X = df[["sleep", "hours"]]\nmodel = LinearRegression()\nmodel.fit(X, y)'), "wrong", "order"),
    ("mean-model", _m("mean_model.fit(X, y)\nmse_mean = (y - mean_model.predict(X)) ** 2"), "wrong", "no-mean"),
    ("fit-students", _F_START, "empty", None),
    ("fit-students", _F, "correct", None),
    ("fit-students", _f('X = df[["hours", "sleep"]]\nmodel = LinearRegression().fit(X, y)'), "correct", None),
    ("fit-students", _f('X = df.drop(columns="grade")\nmodel = LinearRegression()\nmodel.fit(X, y)'), "correct", None),
    ("fit-students", _f('X = df["hours"]\nmodel = LinearRegression()\nmodel.fit(X, y)'), "error", "one-bracket"),
    ("fit-students", _f('X = df["hours", "sleep"]\nmodel = LinearRegression()\nmodel.fit(X, y)'), "error", "two-names"),
    ("fit-students", _f('X = df[["hours", "sleep"]]\nmodel = LinearRegression()'), "error", "no-fit"),
    ("fit-students", _f('X = df[["hours", "sleep", "grade"]]\nmodel = LinearRegression()\nmodel.fit(X, y)'), "wrong", "with-label"),
    ("fit-students", _f('X = df[["hours"]]\nmodel = LinearRegression()\nmodel.fit(X, y)'), "wrong", "one-feature"),
    ("mean-model", _M_START, "empty", None),
    ("mean-model", _M, "correct", None),
    ("mean-model", _m("mean_model.fit(X, y)\nmse_mean = ((y - mean_model.predict(X)) ** 2).mean()"), "correct", None),
    ("mean-model", _m("mse_mean = ((y - mean_model.predict(X)) ** 2).mean()"), "error", "no-fit"),
    ("mean-model", _m("mean_model.fit(X, y)\nmse_mean = ((y - model.predict(X)) ** 2).mean()"), "wrong", "same-model"),
    ("mean-model", _m("mean_model.fit(X, y)\nmse_mean = (y - mean_model.predict(X)).mean()"), "wrong", "no-square"),
    ("mean-model", _m("mse_mean = 0.9555555555555553"), "almost", "other-value"),
]
