# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-regression-metrics: the MAE and the RMSE of a weather forecast, and its R². And the picture of
the eight test houses of ml-baselines with the line learned from the training houses, where one house sells for less
and less, beside the MAE and the RMSE for every price of it."""
import math
import re

from everything_ai.check import Code, CodeBox, Mistake

W, B = 152.82, 52.32          # the line learned from the 32 training houses, as in ml-baselines
CHEAPER = list(range(0, 201, 10))   # the slider: how many thousand euros less one test house sells for
MOVED = 1                     # the test house closest to the line: 2.7 thousand euros above it


def _houses():
    """The forty houses of ml-train-test-split, split the same way: the 8 test houses."""
    import numpy as np
    rng = np.random.default_rng(18)
    sizes = 0.5 + 1.5 * rng.random(size=40)
    prices = 60 + 150 * sizes + 80 * (rng.random(size=40) - 0.5)
    order = rng.permutation(40)
    return sizes[order[:32]], prices[order[:32]], sizes[order[32:]], prices[order[32:]]


X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = _houses()


def prices(d: float):
    """The test prices when house MOVED sells for d thousand euros less."""
    y = Y_TEST.copy()
    y[MOVED] = y[MOVED] - d
    return y


def metrics(d: float = 0) -> dict[str, float]:
    """MAE, MSE, RMSE, the variance of the test prices and R² of the line on the 8 test houses, with house MOVED d
    thousand euros cheaper."""
    import numpy as np
    y = prices(d)
    r = y - (W * X_TEST + B)
    mse = float(np.mean(r ** 2))
    var = float(np.mean((y - y.mean()) ** 2))
    return {"residual": float(r[MOVED]), "mae": float(np.mean(abs(r))), "mse": mse, "rmse": math.sqrt(mse),
            "var": var, "r2": 1 - mse / var}


def picture(d: float, labels: dict[str, str]) -> str:
    """Left: the 8 test houses (open), the line, and every residual as a red stick; house MOVED, d thousand euros
    cheaper, larger. Right: the MAE (solid) and the RMSE (dashed) for every stop of the slider, with a dot at d. Drawn as
    SVG text, once for every stop of the slider (L.explore); dragging along the right chart sets d."""
    from everything_ai import chart
    y = prices(d)
    left = chart.Panel((0.6, 2.0), (50, 360), title=labels["houses"], xlabel=labels["x"], ylabel=labels["y"])
    left.line([0.62, 1.98], [W * 0.62 + B, W * 1.98 + B], "pred", 2.4)
    for x, p in zip(X_TEST, y):
        left.line([x, x], [p, W * x + B], "residual", 2.0)
    left.points([v for i, v in enumerate(X_TEST) if i != MOVED], [v for i, v in enumerate(y) if i != MOVED], "data", 4.5, hollow=True)
    left.points([X_TEST[MOVED]], [y[MOVED]], "data", 6.5, hollow=True)
    right = chart.Panel((0, 200), (0, 90), title=labels["metrics"], xlabel=labels["cheaper"], ylabel=labels["euros"],
                        xticks=[0, 50, 100, 150, 200])
    mae = [metrics(v)["mae"] for v in CHEAPER]
    rmse = [metrics(v)["rmse"] for v in CHEAPER]
    right.line(CHEAPER, mae, "loss", 2.2)
    right.line(CHEAPER, rmse, "loss", 2.2, dashed=True)
    right.text(198, rmse[-1] + 7, "RMSE", "ink-2", anchor="end")
    right.text(198, mae[-1] + 7, "MAE", "ink-2", anchor="end")
    m = metrics(d)
    right.points([d], [m["mae"]], "loss", 5.5)
    right.points([d], [m["rmse"]], "loss", 5.5, hollow=True)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=280, drag=1)


# The lesson is done when these are correct. "which-metric" is the choice question in the lesson.
REQUIRED = ["mae-rmse", "r2", "which-metric"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_BOX_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# A forecast of six days: off by 1, 1, 0, 3, 0 and 6 degrees. The MAE is 11 / 6, the MSE 47 / 6, so the RMSE is about
# 2.80. Every known mistake gives another number than the right ones and than each other.
_Y = "y = np.array([12, 15, 9, 10, 14, 18])\n"
_PRED = "y_pred = np.array([13, 14, 9, 13, 14, 12])\n"
_OTHER_PRED = Code("np.array([11, 15, 10, 10, 16, 17])")   # off by 1, 0, 1, 0, 2 and 1: the MAE is 5 / 6, the MSE 7 / 6

_M_HEAD = "import math\nimport numpy as np\n\n" + _Y + _PRED + "residuals = y - y_pred\n"
_M_START = _M_HEAD + "mae = ...\nrmse = ...\nmae, rmse\n"
_M_MAE = "mae = np.mean(abs(residuals))\n"
_M_RMSE = "rmse = math.sqrt(np.mean(residuals ** 2))\n"
_M = _M_HEAD + _M_MAE + _M_RMSE + "mae, rmse"


def _m(mae: str = _M_MAE, rmse: str = _M_RMSE, last: str = "mae, rmse") -> str:
    return _M_HEAD + mae + rmse + last


def _numbers(got) -> list[float] | None:
    try:
        return [float(v) for v in got]
    except (TypeError, ValueError):
        return None


def _swapped(got, text) -> bool:
    """The two right numbers, the other way round: first the RMSE."""
    v = _numbers(got)
    return v is not None and len(v) == 2 and abs(v[0] - math.sqrt(47 / 6)) < 1e-9 and abs(v[1] - 11 / 6) < 1e-9


def _rounded(got, text) -> bool:
    """Both numbers close to the right ones, but at least one of them rounded or typed in."""
    v = _numbers(got)
    want = (11 / 6, math.sqrt(47 / 6))
    return (v is not None and len(v) == 2 and all(abs(a - b) <= 0.006 for a, b in zip(v, want))
            and any(abs(a - b) > 1e-9 for a, b in zip(v, want)))


def _in_a_list(got, text) -> bool:
    """The two right numbers, in a list."""
    v = _numbers(got)
    return isinstance(got, list) and v is not None and len(v) == 2 and abs(v[0] - 11 / 6) < 1e-9 and abs(v[1] - math.sqrt(47 / 6)) < 1e-9


# The same forecast. The variance of the six temperatures is 56 / 6, so R² is 1 - 47 / 56, about 0.161.
_R_HEAD = "import numpy as np\n\n" + _Y + _PRED + "mse = np.mean((y - y_pred) ** 2)\n"
_R_START = _R_HEAD + "variance = ...\nr2 = ...\nr2\n"
_R_VAR = "variance = np.mean((y - y.mean()) ** 2)\n"
_R_R2 = "r2 = 1 - mse / variance\n"
_R = _R_HEAD + _R_VAR + _R_R2 + "r2"


def _r(var: str = _R_VAR, r2: str = _R_R2) -> str:
    return _R_HEAD + var + r2 + "r2"


def _near(value: float, tol: float):
    """A number close to `value` but not equal to it, such as R² rounded or typed in."""
    def matches(got, text) -> bool:
        try:
            return 1e-9 < abs(float(got) - value) <= tol
        except (TypeError, ValueError):
            return False
    return matches


EXERCISES = {
    "mae-rmse": CodeBox(
        name="mae-rmse",
        start=_M_START,
        solution=_M,
        digits=4,
        # The check also runs the box with another forecast: numbers typed in by hand do not change with it.
        others=[{"y_pred": _OTHER_PRED}],
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", matches=_swapped,
                    en="Your last line gives the two right numbers, but the other way round: first the RMSE. Make the last line `mae, rmse` again: that is how the check reads them.",
                    nl="Je laatste regel geeft de twee goede getallen, maar andersom: eerst de RMSE. Maak van de laatste regel weer `mae, rmse`: zo leest de check ze."),
            Mistake("list", state="almost", matches=_in_a_list,
                    en="Your last line gives the right numbers, but in a list: `{got}`. Make it `mae, rmse` again, without the square brackets: that is how the check reads them.",
                    nl="Je laatste regel geeft de goede getallen, maar in een lijst: `{got}`. Maak er weer `mae, rmse` van, zonder de vierkante haken: zo leest de check ze."),
            Mistake("rounded", state="almost", matches=_rounded,
                    en="Your last line gives `{got}`: close, but rounded. Let Python work out the MAE and the RMSE from `residuals`, without rounding and without typing in a number.",
                    nl="Je laatste regel geeft `{got}`: dichtbij, maar afgerond. Laat Python de MAE en de RMSE uitrekenen uit `residuals`, zonder afronden en zonder zelf een getal in te typen."),
            Mistake("no-abs", wrong=_m(mae="mae = np.mean(residuals)\n"),
                    en="Your last line gives `{got}`: your `mae` is the mean residual, with its sign, so the misses up and down cancel. Take the size of each residual first: `abs(residuals)`.",
                    nl="Je laatste regel geeft `{got}`: je `mae` is het gemiddelde residu, met het teken erbij, dus de missers omhoog en omlaag vallen tegen elkaar weg. Neem eerst de grootte van elk residu: `abs(residuals)`."),
            Mistake("no-root", wrong=_m(rmse="rmse = np.mean(residuals ** 2)\n"),
                    en="Your last line gives `{got}`: your `rmse` is the MSE, in degrees squared. Take its square root, with `math.sqrt`, to get back to degrees.",
                    nl="Je laatste regel geeft `{got}`: je `rmse` is de MSE, in graden in het kwadraat. Neem de wortel eruit, met `math.sqrt`, om weer in graden uit te komen."),
            Mistake("root-each", wrong=_m(rmse="rmse = np.mean((residuals ** 2) ** 0.5)\n"),
                    en="Your last line gives `{got}`: your `rmse` takes the root of each square before the mean, and that gives the MAE again. First the mean of the squares, then one square root: `math.sqrt(np.mean(residuals ** 2))`.",
                    nl="Je laatste regel geeft `{got}`: je `rmse` neemt de wortel van elk kwadraat vóór het gemiddelde, en dan krijg je weer de MAE. Eerst het gemiddelde van de kwadraten, dan één wortel: `math.sqrt(np.mean(residuals ** 2))`."),
            Mistake("root-of-mae", wrong=_m(rmse="rmse = math.sqrt(np.mean(abs(residuals)))\n"),
                    en="Your last line gives `{got}`: your `rmse` is the square root of the MAE. The RMSE squares the residuals first: the root of the mean of `residuals ** 2`.",
                    nl="Je laatste regel geeft `{got}`: je `rmse` is de wortel uit de MAE. De RMSE kwadrateert eerst de residuen: de wortel uit het gemiddelde van `residuals ** 2`."),
            Mistake("std-residuals", wrong=_m(rmse="rmse = np.std(residuals)\n"),
                    en="Your last line gives `{got}`: your `rmse` is the standard deviation of the residuals, which measures them around their own mean. The RMSE measures how far they are from 0: the square root of the mean of `residuals ** 2`.",
                    nl="Je laatste regel geeft `{got}`: je `rmse` is de standaardafwijking van de residuen, en die meet ze rond hun eigen gemiddelde. De RMSE meet hoe ver ze van 0 af liggen: de wortel uit het gemiddelde van `residuals ** 2`."),
            Mistake("sum", wrong=_m(mae="mae = np.sum(abs(residuals))\n"),
                    en="Your last line gives `{got}`: your `mae` is the sum of the sizes of the residuals, not their mean. Take the mean: `np.mean`.",
                    nl="Je laatste regel geeft `{got}`: je `mae` is de som van de groottes van de residuen, niet hun gemiddelde. Neem het gemiddelde: `np.mean`."),
        ],
        texts=_BOX_TEXTS | {
            "empty": {"en": "Replace the two `...`: the MAE in `mae`, and the RMSE in `rmse`.",
                      "nl": "Vervang de twee `...`: de MAE in `mae`, en de RMSE in `rmse`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `mae, rmse` on the last line, so the check can see them.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `mae, rmse` op de laatste regel, zodat de check ze kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `mae` the mean of `abs(residuals)`, and `rmse` the square root of the mean of `residuals ** 2`?",
                      "nl": "Je laatste regel geeft `{got}`. Is `mae` het gemiddelde van `abs(residuals)`, en `rmse` de wortel uit het gemiddelde van `residuals ** 2`?"},
            "other": {"en": "Right for this forecast. But with {where}, your last line gives `{got}`, and it should give about `{want}`. Do you work them out from `residuals` itself?",
                      "nl": "Goed voor deze voorspelling. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `{want}` zijn. Reken je ze uit met `residuals` zelf?"},
            "other_error": {"en": "Right for this forecast. But with {where}, your code stops with an error. Do you work them out from `residuals` itself?",
                            "nl": "Goed voor deze voorspelling. Maar met {where} stopt je code met een fout. Reken je ze uit met `residuals` zelf?"},
            "correct": {"en": "An MAE of about 1.83 degrees and an RMSE of about 2.80: the forecast is typically about 2 degrees off, and the RMSE is well above the MAE because of the one day it was 6 degrees off. The check also tried another forecast.",
                        "nl": "Een MAE van ongeveer 1,83 graden en een RMSE van ongeveer 2,80: de voorspelling zit meestal zo'n 2 graden ernaast, en de RMSE ligt een flink stuk boven de MAE door die ene dag met 6 graden ernaast. De check probeerde ook een andere voorspelling."},
        },
    ),
    "r2": CodeBox(
        name="r2",
        start=_R_START,
        solution=_R,
        digits=4,
        # The check also runs the box with another forecast: an R² typed in does not change with it.
        others=[{"y_pred": _OTHER_PRED}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-one-minus", wrong=_r(r2="r2 = mse / variance\n"),
                    en="Your last line gives `{got}`: the MSE divided by the variance, the share the forecast does not explain. R² is 1 minus that: `1 - mse / variance`.",
                    nl="Je laatste regel geeft `{got}`: de MSE gedeeld door de variantie, het deel dat de voorspelling niet verklaart. R² is 1 min dat: `1 - mse / variance`."),
            Mistake("flipped", wrong=_r(r2="r2 = 1 - variance / mse\n"),
                    en="Your last line gives `{got}`: your `r2` divides the variance by the MSE, the other way round. R² is 1 minus the MSE divided by the variance: `1 - mse / variance`.",
                    nl="Je laatste regel geeft `{got}`: je `r2` deelt de variantie door de MSE, andersom dus. R² is 1 min de MSE gedeeld door de variantie: `1 - mse / variance`."),
            Mistake("pred-variance", wrong=_r(var="variance = np.mean((y_pred - y_pred.mean()) ** 2)\n"),
                    en="Your last line gives `{got}`: your `variance` is the variance of the predictions. R² compares with the variance of the labels themselves, `y`: how far they spread around their own mean.",
                    nl="Je laatste regel geeft `{got}`: je `variance` is de variantie van de voorspellingen. R² vergelijkt met de variantie van de labels zelf, `y`: hoe ver die rond hun eigen gemiddelde liggen."),
            Mistake("std", wrong=_r(var="variance = np.std(y)\n"),
                    en="Your last line gives `{got}`: your `variance` is the standard deviation, in degrees. The variance is its square, in degrees squared like the MSE: the mean of `(y - y.mean()) ** 2`.",
                    nl="Je laatste regel geeft `{got}`: je `variance` is de standaardafwijking, in graden. De variantie is het kwadraat daarvan, in graden in het kwadraat, net als de MSE: het gemiddelde van `(y - y.mean()) ** 2`."),
            Mistake("mean-pred", wrong=_r(var="variance = np.mean((y - y_pred.mean()) ** 2)\n"),
                    en="Your last line gives `{got}`: your `variance` measures the labels around the mean of the predictions. Measure them around their own mean, `y.mean()`.",
                    nl="Je laatste regel geeft `{got}`: je `variance` meet de labels rond het gemiddelde van de voorspellingen. Meet ze rond hun eigen gemiddelde, `y.mean()`."),
            Mistake("no-square", wrong=_r(var="variance = np.mean(y - y.mean())\n"),
                    en="Your `variance` is the mean distance to the mean, with its sign, and that is always 0, so `mse` is divided by 0. Square each distance first: `** 2`.",
                    nl="Je `variance` is de gemiddelde afstand tot het gemiddelde, met het teken erbij, en die is altijd 0, dus `mse` wordt door 0 gedeeld. Kwadrateer eerst elke afstand: `** 2`."),
            Mistake("rounded", state="almost", matches=_near(1 - 47 / 56, 0.006),
                    en="Your last line gives `{got}`: close, but rounded. Let Python work out R² from `mse` and `variance`, without rounding and without typing in a number.",
                    nl="Je laatste regel geeft `{got}`: dichtbij, maar afgerond. Laat Python R² uitrekenen uit `mse` en `variance`, zonder afronden en zonder zelf een getal in te typen."),
        ],
        texts=_BOX_TEXTS | {
            "empty": {"en": "Replace the two `...`: the variance of the six temperatures in `variance`, and R² in `r2`.",
                      "nl": "Vervang de twee `...`: de variantie van de zes temperaturen in `variance`, en R² in `r2`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `r2` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `r2` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `variance` the mean of `(y - y.mean()) ** 2`, and `r2` equal to 1 minus `mse` divided by it?",
                      "nl": "Je laatste regel geeft `{got}`. Is `variance` het gemiddelde van `(y - y.mean()) ** 2`, en is `r2` 1 min `mse` gedeeld door die variantie?"},
            "other": {"en": "Right for this forecast. But with {where}, your last line gives `{got}`, and it should give about `{want}`. Do you work it out from `y`, `mse` and `variance` themselves?",
                      "nl": "Goed voor deze voorspelling. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `{want}` zijn. Reken je het uit met `y`, `mse` en `variance` zelf?"},
            "other_error": {"en": "Right for this forecast. But with {where}, your code stops with an error. Do you work it out from `y`, `mse` and `variance` themselves?",
                            "nl": "Goed voor deze voorspelling. Maar met {where} stopt je code met een fout. Reken je het uit met `y`, `mse` en `variance` zelf?"},
            "correct": {"en": "An R² of about 0.161: the forecast explains only about a sixth of the variance of the six temperatures. Its MAE of under 2 degrees sounded good, but always saying 13 degrees, their mean, has an MSE of 9.33 against its 7.83: not much worse. The check also tried another forecast.",
                        "nl": "Een R² van ongeveer 0,161: de voorspelling verklaart maar zo'n zesde van de variantie van de zes temperaturen. Zijn MAE van minder dan 2 graden klonk goed, maar elke dag 13 graden zeggen, hun gemiddelde, heeft een MSE van 9,33 tegen zijn 7,83: niet veel slechter. De check probeerde ook een andere voorspelling."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("mae-rmse", _M_START, "empty", None),
    ("mae-rmse", _M, "correct", None),
    ("mae-rmse", _m(mae="mae = np.mean(np.abs(y - y_pred))\n", rmse="rmse = np.mean(residuals ** 2) ** 0.5\n"), "correct", None),
    ("mae-rmse", _m(mae="mae = sum(abs(residuals)) / len(residuals)\n", rmse="mse = np.mean(residuals ** 2)\nrmse = math.sqrt(mse)\n"), "correct", None),
    ("mae-rmse", _m(last="rmse, mae"), "wrong", "swapped"),
    ("mae-rmse", _m(last="[mae, rmse]"), "almost", "list"),
    ("mae-rmse", _m(rmse="rmse = round(math.sqrt(np.mean(residuals ** 2)), 2)\n"), "almost", "rounded"),
    ("mae-rmse", _m(mae="mae = np.mean(residuals)\n"), "wrong", "no-abs"),
    ("mae-rmse", _m(rmse="rmse = np.mean(residuals ** 2)\n"), "wrong", "no-root"),
    ("mae-rmse", _m(rmse="rmse = np.mean(np.sqrt(residuals ** 2))\n"), "wrong", "root-each"),
    ("mae-rmse", _m(rmse="rmse = math.sqrt(np.mean(abs(residuals)))\n"), "wrong", "root-of-mae"),
    ("mae-rmse", _m(mae="mae = np.sum(abs(residuals))\n"), "wrong", "sum"),
    ("mae-rmse", _m(rmse="rmse = np.std(residuals)\n"), "wrong", "std-residuals"),
    ("mae-rmse", _m(mae="mae = 11 / 6\n", rmse="rmse = math.sqrt(47 / 6)\n"), "almost", "other-value"),
    ("mae-rmse", _m(mae="mae = ...\n"), "wrong", "dots-left"),
    ("r2", _R_START, "empty", None),
    ("r2", _R, "correct", None),
    ("r2", _r(var="variance = np.std(y) ** 2\n", r2="r2 = 1 - (mse / variance)\n"), "correct", None),
    ("r2", _r(var="baseline = y.mean()\nvariance = np.mean((y - baseline) ** 2)\n"), "correct", None),
    ("r2", _r(r2="r2 = mse / variance\n"), "wrong", "no-one-minus"),
    ("r2", _r(r2="r2 = 1 - variance / mse\n"), "wrong", "flipped"),
    ("r2", _r(var="variance = np.mean((y_pred - y_pred.mean()) ** 2)\n"), "wrong", "pred-variance"),
    ("r2", _r(var="variance = np.std(y)\n"), "wrong", "std"),
    ("r2", _r(var="variance = np.mean((y - y_pred.mean()) ** 2)\n"), "wrong", "mean-pred"),
    ("r2", _r(var="variance = np.mean(y - y.mean())\n"), "wrong", "no-square"),
    ("r2", _r(r2="r2 = round(1 - mse / variance, 2)\n"), "almost", "rounded"),
    ("r2", _r(r2="r2 = 1 - 47 / 56\n"), "almost", "other-value"),
    ("r2", _r(var="variance = ...\n"), "wrong", "dots-left"),
]
