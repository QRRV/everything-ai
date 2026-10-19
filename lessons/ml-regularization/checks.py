# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-regularization: the loss with an L2 penalty, and its gradients, which pull every weight
toward 0. And the picture of the eight weights and the training and validation MSE as λ moves on the slider."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, same_value

SEED, N, D, N_TRAIN = 2, 40, 8, 12   # as in the worked example: 12 examples to train on, 28 to validate on
LAMBDAS = [round(0.01 * k, 2) for k in range(21)]   # the slider: λ from 0 to 0.2
# The lines: λ every 0.001. The validation MSE drops steeply between λ = 0 and 0.01, to its lowest near 0.006;
# a line through the slider's 21 values alone would draw that drop as a straight line, far above the real curve.
CURVE = [round(0.001 * k, 3) for k in range(201)]


def _data():
    import numpy as np
    rng = np.random.default_rng(SEED)
    X = rng.random((N, D))
    y = 1 + 3 * X[:, 0] - 2 * X[:, 1] + (rng.random(N) - 0.5)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = _data()


def fit(lam: float):
    """The weights and bias term with the lowest regularized loss on the training set: where gradient descent with
    weight decay ends after very many steps, worked out directly so that the picture follows the slider at once."""
    import numpy as np
    xm, ym = X_TRAIN.mean(axis=0), Y_TRAIN.mean()
    xc, yc = X_TRAIN - xm, Y_TRAIN - ym
    w = np.linalg.solve(xc.T @ xc / N_TRAIN + lam * np.eye(D), xc.T @ yc / N_TRAIN)
    return w, float(ym - xm @ w)


def errors(lam: float) -> tuple[float, float]:
    """The MSE on the training set and on the validation set for this λ."""
    import numpy as np
    w, b = fit(lam)
    return (float(np.mean((Y_TRAIN - (X_TRAIN @ w + b)) ** 2)), float(np.mean((Y_VAL - (X_VAL @ w + b)) ** 2)))


ERRORS = {lam: errors(lam) for lam in sorted(set(LAMBDAS) | set(CURVE))}


def picture(lam: float, labels: dict[str, str]) -> str:
    """The eight weights as bars, with a dash at the weight the labels were made with (3, −2 and six 0s); and the
    training MSE (filled) and validation MSE (open) for every λ, with the slider's λ larger. Dragging across that
    chart sets λ."""
    from everything_ai import chart
    lam = min(LAMBDAS, key=lambda v: abs(v - lam))
    w, _ = fit(lam)
    bars = chart.Panel((0.4, 8.6), (-4, 4), title=labels["weights"], xlabel=labels["feature"], ylabel=labels["weight"],
                       xticks=list(range(1, 9)))
    bars.line([0.4, 8.6], [0, 0], "ink-3", 1.2)
    for j, (wj, true) in enumerate(zip(w, [3, -2, 0, 0, 0, 0, 0, 0])):
        bars.rect(j + 1 - 0.3, 0, j + 1 + 0.3, max(-4, min(4, wj)), "param")
        bars.line([j + 1 - 0.38, j + 1 + 0.38], [true, true], "ink", 2.0, dashed=True)
    mse = chart.Panel((0, 0.2), (0, 1.4), title=labels["mse"], xlabel="λ", ylabel=labels["loss"])
    train = [ERRORS[v][0] for v in CURVE]
    val = [ERRORS[v][1] for v in CURVE]
    mse.line(CURVE, train, "loss", 2.0)
    mse.line(CURVE, val, "loss", 2.0, dashed=True)
    mse.points([lam], [ERRORS[lam][0]], "loss", 6)
    mse.points([lam], [ERRORS[lam][1]], "loss", 6, hollow=True)
    return chart.figure([bars, mse], alt=labels["alt"], width=330, height=300, drag=1)   # two fit side by side in a play block; drag along λ


# The lesson is done when these are correct. "big-lambda" is the choice question in the lesson.
REQUIRED = ["big-lambda", "penalty-loss", "decay-gradients"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_L_HEAD = ("import numpy as np\n\nX = np.array([[0.2, 0.9], [0.5, 0.4], [0.8, 0.1], [0.3, 0.7]])\ny = np.array([0.5, 1.7, 3.0, 1.1])\n"
           "w = np.array([2.0, -1.5])\nb = 1.0\nlam = 0.1\n")
_L_START = _L_HEAD + "loss = ...\nloss\n"
_L_LINE = "loss = np.mean((y - (X @ w + b)) ** 2) + lam * np.sum(w ** 2)"
_L = _L_HEAD + _L_LINE + "\nloss"


def _l(new: str) -> str:
    return _L.replace(_L_LINE, new)


_G_HEAD = "import numpy as np\n\n\ndef decay_gradients(X, y, w, b, lam):\n    X = np.array(X)\n    y = np.array(y)\n    w = np.array(w)\n    r = y - (X @ w + b)\n"
_G_START = _G_HEAD + "    ...\n    return grad_w, grad_b\n"
_G_BODY = "    grad_w = -2 * (r @ X) / len(y) + 2 * lam * w\n    grad_b = -2 * r.mean()\n    return grad_w, grad_b"
_G = _G_HEAD + _G_BODY


def _g(grad_w: str, grad_b: str = "-2 * r.mean()") -> str:
    return _G_HEAD + f"    grad_w = {grad_w}\n    grad_b = {grad_b}\n    return grad_w, grad_b"


def _number(x: float) -> str:
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _pair(v):
    """What the function gives back, with numbers to four decimals: a tuple as (..), a list or array as [..], so
    the pair (slopes for the weights, slope for b) shows as ([..], ..); anything else as Python shows it."""
    import numpy as np
    if isinstance(v, tuple):
        return "(" + ", ".join(_pair(x) for x in v) + ")"
    if isinstance(v, list) or isinstance(v, np.ndarray) and v.ndim > 0:
        return "[" + ", ".join(_pair(x) for x in v) + "]"
    try:
        return _number(float(v))
    except (TypeError, ValueError):
        return repr(v)


def _same_pair(got, want) -> bool:
    """The two slopes as a list, [grad_w, grad_b], are the same outcome as the pair (grad_w, grad_b)."""
    return same_value(tuple(got) if isinstance(got, list) else got, want)


_SMALL = ([[0.2, 0.9], [0.5, 0.4], [0.8, 0.1], [0.3, 0.7]], [0.5, 1.7, 3.0, 1.1], [2.0, -1.5], 1.0, 0.1)

EXERCISES = {
    "penalty-loss": CodeBox(
        name="penalty-loss",
        start=_L_START,
        solution=_L,
        keep=["w = np.array([2.0, -1.5])", "lam = 0.1"],
        # The check also runs the box with other weights: a number typed in by hand does not change with them.
        others=[{"w": Code("np.array([1.0, 0.5])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-penalty", wrong=_l("loss = np.mean((y - (X @ w + b)) ** 2)"),
                    en="Your last line gives `{got}`: the mean squared error alone. Add the penalty: `lam` times the sum of the squared weights.",
                    nl="Je laatste regel geeft `{got}`: alleen de mean squared error. Tel de straf erbij op: `lam` keer de som van de gekwadrateerde weights."),
            Mistake("with-bias", wrong=_l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * (np.sum(w ** 2) + b ** 2)"),
                    en="Your last line gives `{got}`: you also punish the bias term. The penalty is only for the weights: `lam * np.sum(w ** 2)`.",
                    nl="Je laatste regel geeft `{got}`: je straft ook de bias term. De straf is alleen voor de weights: `lam * np.sum(w ** 2)`."),
            Mistake("not-squared", wrong=_l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * np.sum(w)"),
                    en="Your last line gives `{got}`: you add the weights themselves, and a negative weight then lowers the loss. Square each weight first: `w ** 2`.",
                    nl="Je laatste regel geeft `{got}`: je telt de weights zelf op, en een negatieve weight verlaagt dan de loss. Kwadrateer eerst elke weight: `w ** 2`."),
            Mistake("mean-of-weights", wrong=_l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * np.mean(w ** 2)"),
                    en="Your last line gives `{got}`: the penalty is half as big as it should be, as with the mean of the two squared weights. The penalty is `lam` times their sum: `lam * np.sum(w ** 2)`.",
                    nl="Je laatste regel geeft `{got}`: de straf is half zo groot als hij moet zijn, zoals met het gemiddelde van de twee gekwadrateerde weights. De straf is `lam` keer hun som: `lam * np.sum(w ** 2)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the regularized loss: the mean squared error plus `lam` times the sum of the squared weights.",
                      "nl": "Vervang de `...` door de geregulariseerde loss: de mean squared error plus `lam` keer de som van de gekwadrateerde weights."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other weights by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere weights."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is it the mean squared error of `X @ w + b`, plus `lam` times the sum of the squared weights?",
                      "nl": "Je laatste regel geeft `{got}`. Is het de mean squared error van `X @ w + b`, plus `lam` keer de som van de gekwadrateerde weights?"},
            "other": {"en": "Right for these weights. But with {where}, your last line gives `{got}`, and it should give about `0.86`. Do you work it out with `w` itself?",
                      "nl": "Goed voor deze weights. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `0.86` zijn. Reken je het uit met `w` zelf?"},
            "other_error": {"en": "Right for these weights. But with {where}, your code stops with an error. Do you work it out with `w` itself?",
                            "nl": "Goed voor deze weights. Maar met {where} stopt je code met een fout. Reken je het uit met `w` zelf?"},
            "correct": {"en": "About 0.85: a mean squared error of about 0.22, plus a penalty of 0.625: 0.1 times 6.25, the sum of the squares of 2 and −1.5. The check also tried other weights.",
                        "nl": "Ongeveer 0,85: een mean squared error van ongeveer 0,22, plus een straf van 0,625: 0,1 keer 6,25, de som van de kwadraten van 2 en −1,5. De check probeerde ook andere weights."},
        },
    ),
    "decay-gradients": FunctionBox(
        name="decay-gradients",
        function="decay_gradients",
        start=_G_START,
        solution=_G,
        visible=[_SMALL],
        hidden=[([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 2.0, 2.5], [0.0, 0.0], 0.0, 0.5),
                ([[0.5], [1.5], [2.5]], [1.0, 2.0, 4.0], [1.2], 0.3, 0.0),
                ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [1.0, 0.0], [1.0, -1.0, 2.0], -0.5, 1.0)],
        show=_pair,
        compare=_same_pair,
        first=[_DOTS],
        mistakes=[
            Mistake("no-decay", wrong=_g("-2 * (r @ X) / len(y)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the slopes of the mean squared error alone. The penalty adds `2 * lam * w` to the slopes for the weights.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: alleen de hellingen van de mean squared error. De straf telt `2 * lam * w` op bij de hellingen voor de weights."),
            Mistake("bias-decay", wrong=_g("-2 * (r @ X) / len(y) + 2 * lam * w", "-2 * r.mean() + 2 * lam * b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you pull the bias term toward 0 as well. The penalty is only for the weights, so the slope for `b` stays `-2 * r.mean()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je trekt ook de bias term naar 0. De straf is alleen voor de weights, dus de helling voor `b` blijft `-2 * r.mean()`."),
            Mistake("no-two", wrong=_g("-2 * (r @ X) / len(y) + lam * w"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the slope of `lam * w ** 2` for a weight is `2 * lam * w`, with a 2, as the slope of a square always has.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de helling van `lam * w ** 2` voor een weight is `2 * lam * w`, met een 2, zoals de helling van een kwadraat altijd heeft."),
            Mistake("square", wrong=_g("-2 * (r @ X) / len(y) + lam * w ** 2"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add the penalty itself, `lam * w ** 2`. The slopes need the slope of the penalty, not the penalty itself: `2 * lam * w`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de straf zelf op, `lam * w ** 2`. De hellingen hebben de helling van de straf nodig, niet de straf zelf: `2 * lam * w`."),
            Mistake("minus", wrong=_g("-2 * (r @ X) / len(y) - 2 * lam * w"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the wrong sign on the penalty. As a positive weight grows, the penalty grows, so its slope is `+ 2 * lam * w`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het verkeerde teken bij de straf. Als een positieve weight groeit, groeit de straf mee, dus zijn helling is `+ 2 * lam * w`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with `grad_w` and `grad_b`: the slopes of the mean squared error, and for the weights also the slope of the penalty.",
                      "nl": "Vervang de `...` door `grad_w` en `grad_b`: de hellingen van de mean squared error, en voor de weights ook de helling van de straf."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, w, b, lam)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, w, b, lam)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Are the slopes for the weights the slopes of the mean squared error plus the slope of the penalty, and the slope for `b` that of the mean squared error alone?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zijn de hellingen voor de weights de hellingen van de mean squared error plus de helling van de straf, en is de helling voor `b` alleen die van de mean squared error?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `X`, `y`, `w`, `b` and `lam` themselves, for any number of examples and features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X`, `y`, `w`, `b` en `lam` zelf, voor elk aantal voorbeelden en features?"},
            "correct": {"en": "Your function is right in all {n} cases. With `lam = 0` it gives the slopes of plain linear regression; every bit of `lam` pulls each weight a little more toward 0.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met `lam = 0` geeft hij de hellingen van gewone lineaire regressie; elk beetje `lam` trekt elke weight iets meer naar 0."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("penalty-loss", _L_START, "empty", None),
    ("penalty-loss", _L, "correct", None),
    ("penalty-loss", _l("loss = ((y - X @ w - b) ** 2).mean() + lam * (w ** 2).sum()"), "correct", None),
    ("penalty-loss", _l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * (w @ w)"), "correct", None),
    ("penalty-loss", _l("loss = np.mean((y - (X @ w + b)) ** 2)"), "wrong", "no-penalty"),
    ("penalty-loss", _l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * (np.sum(w ** 2) + b ** 2)"), "wrong", "with-bias"),
    ("penalty-loss", _l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * np.sum(w)"), "wrong", "not-squared"),
    ("penalty-loss", _l("loss = np.mean((y - (X @ w + b)) ** 2) + lam * np.mean(w ** 2)"), "wrong", "mean-of-weights"),
    ("penalty-loss", _l("loss = np.mean((y - (X @ w + b)) ** 2) + lam / 2 * np.sum(w ** 2)"), "wrong", "mean-of-weights"),
    ("penalty-loss", _l("loss = 0.849375"), "almost", "other-value"),
    ("penalty-loss", _L.replace("w = np.array([2.0, -1.5])", "w = np.array([1.0, 0.5])"), "wrong", "changed-line"),
    ("decay-gradients", _G_START, "empty", None),
    ("decay-gradients", _G, "correct", None),
    ("decay-gradients", _G_HEAD + "    grad_w = -2 * X.T @ r / len(y) + 2 * lam * w\n    grad_b = -2 * np.mean(r)\n    return grad_w, grad_b", "correct", None),
    ("decay-gradients", _G_HEAD + "    grad_w = 2 * ((X @ w + b - y) @ X) / len(y) + 2 * lam * w\n    grad_b = 2 * np.mean(X @ w + b - y)\n    return grad_w, grad_b", "correct", None),
    ("decay-gradients", _G_HEAD + "    grad_w = -2 * (r @ X) / len(y) + 2 * lam * w\n    grad_b = -2 * r.mean()\n    return [grad_w, grad_b]", "correct", None),
    ("decay-gradients", _g("-2 * (r @ X) / len(y)"), "wrong", "no-decay"),
    ("decay-gradients", _g("-2 * (r @ X) / len(y) + 2 * lam * w", "-2 * r.mean() + 2 * lam * b"), "wrong", "bias-decay"),
    ("decay-gradients", _g("-2 * (r @ X) / len(y) + lam * w"), "wrong", "no-two"),
    ("decay-gradients", _g("-2 * (r @ X) / len(y) + lam * w ** 2"), "wrong", "square"),
    ("decay-gradients", _g("-2 * (r @ X) / len(y) - 2 * lam * w"), "wrong", "minus"),
    ("decay-gradients", _g("-2 * (r @ X) / 4 + 2 * lam * w"), "almost", None),
]
