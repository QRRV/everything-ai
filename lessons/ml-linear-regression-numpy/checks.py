# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson ml-linear-regression-numpy: mse_gradients and fit, each saved in the learner's
project (work/projects/nn/) when right, and the demo that trains the learner's own model on eight houses."""
import math
import re

from everything_ai import projects
from everything_ai.check import FunctionBox, Mistake, run_code

# Eight houses: area in hundreds of m², a garden (1 = yes, 0 = no), and the distance to the station in km. The price
# is in thousands of euros. The units keep the numbers of the three features about equally big.
HOUSES = [[0.62, 0, 0.4], [0.85, 1, 1.2], [1.10, 1, 0.8], [0.48, 0, 0.3],
          [0.95, 0, 1.5], [1.30, 1, 0.6], [0.73, 1, 0.9], [0.58, 0, 1.1]]
PRICES = [191, 218, 290, 162, 215, 333, 211, 150]
LR, STEPS = 0.2, 1000

# The small example of the picture: r @ X is the dot product of r with each column of X.
R_SMALL = [2, -1, 1, 3]
X_SMALL = [[1, 0], [2, 1], [0, 2], [1, 1]]


def _slopes(X, y, w, b):
    """The course's own gradient of the MSE, so that a wrong module of the learner shows up."""
    import numpy as np
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    r = y - (X @ np.array(w, dtype=float) + b)
    return -2 * (r @ X) / len(y), -2 * r.mean()


def _fit(X, y, lr, steps):
    import numpy as np
    X = np.array(X, dtype=float)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        gw, gb = _slopes(X, y, w, b)
        w, b = w - lr * gw, b - lr * gb
    return w, b


def _close_pair(got, want) -> bool:
    """(slopes or weights, one number for b), the same as the course's within a hair."""
    try:
        gw, gb = got
        ww, wb = want
        gw, ww = [float(v) for v in gw], [float(v) for v in ww]
        gb, wb = float(gb), float(wb)
    except (TypeError, ValueError, OverflowError):
        return False
    return (len(gw) == len(ww) and all(math.isclose(g, v, rel_tol=1e-6, abs_tol=1e-9) for g, v in zip(gw, ww))
            and math.isclose(gb, wb, rel_tol=1e-6, abs_tol=1e-9))


def _pair(v) -> str:
    """A pair (array, number) as the lessons write it: ([-2.0, -6.0], -4.0), rounded to 4 decimals."""
    try:
        w, b = v
        return f"({[round(float(x), 4) for x in w]}, {round(float(b), 4)})"
    except (TypeError, ValueError, OverflowError):
        return _plain(v)


def _plain(v) -> str:
    """Anything else as the lessons write values: lists and numbers, never array(...) or np.float64(...)."""
    import numpy as np
    if isinstance(v, (tuple, list)):
        inner = ", ".join(_plain(x) for x in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    try:
        return str(np.round(np.asarray(v, dtype=float), 4).tolist())
    except (TypeError, ValueError):
        return repr(v)


def gradients_ok() -> bool | None:
    """None when mse_gradients is not saved yet; else whether the saved one still gives the right slopes."""
    if not (projects.folder() / "nn" / "gradients.py").exists():
        return None
    for case in [([[1.0, 2.0], [3.0, 0.0], [0.0, 1.0]], [3.0, 0.0, 3.0], [0.5, -1.0], 0.25), (HOUSES, PRICES, [200, 20, -40], 80)]:
        run = run_code(f"from nn import gradients\ngradients.mse_gradients{case!r}")
        if run.error is not None or run.timeout or not _close_pair(run.value, _slopes(*case)):
            return False
    return True


def trained():
    """For the demo: None when fit is not saved yet; False when the saved fit no longer trains right, or stops with an
    error; else (w, b) after the learner's own fit on the houses."""
    if not (projects.folder() / "nn" / "linear.py").exists():
        return None
    if gradients_ok() is not True:
        return "gradients"
    run = run_code(f"from nn import linear\nlinear.fit({HOUSES!r}, {PRICES!r}, {LR}, {STEPS})")
    if run.error is not None or not _close_pair(run.value, _fit(HOUSES, PRICES, LR, STEPS)):
        return False
    w, b = run.value
    return [float(v) for v in w], float(b)


def demo_picture(w: list, b: float, labels: dict[str, str]) -> str:
    """Every house as a point: its real price across, the prediction of the learner's model up. On the dashed line
    the prediction is exactly right."""
    from everything_ai import chart
    preds = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in HOUSES]
    p = chart.Panel((140, 350), (140, 350), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line([140, 350], [140, 350], "ink-3", 1.4, dashed=True)
    p.points(PRICES, preds, "pred", 5)
    return chart.figure([p], alt=labels["alt"], width=420, height=380)


# The lesson is done when these are correct. "slope-shape" is the choice question in the lesson.
REQUIRED = ["slope-shape", "mse-gradients", "fit"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")

_GRAD_HEAD = "import numpy as np\n\n\ndef mse_gradients(X, y, w, b):\n    X = np.array(X)\n    y = np.array(y)\n"
_GRAD_START = _GRAD_HEAD + "    ...\n    return grad_w, grad_b\n"
_GRAD_BODY = "    r = y - (X @ w + b)\n    grad_w = -2 * (r @ X) / len(y)\n    grad_b = -2 * r.mean()\n    return grad_w, grad_b"
_GRAD = _GRAD_HEAD + _GRAD_BODY


def _grad(new: str) -> str:
    return _GRAD_HEAD + new


_FIT_HEAD = ("import numpy as np\nfrom nn import gradients\n\n\ndef fit(X, y, lr, steps):\n    X = np.array(X)\n    y = np.array(y)\n"
             "    w = np.zeros(X.shape[1])\n    b = 0.0\n    for step in range(steps):\n")
_FIT_START = _FIT_HEAD + "        ...\n    return w, b\n"
_FIT_BODY = ("        grad_w, grad_b = gradients.mse_gradients(X, y, w, b)\n        w = w - lr * grad_w\n        b = b - lr * grad_b\n"
             "    return w, b")
_FIT = _FIT_HEAD + _FIT_BODY
# What the check compares with: the course's own slopes, not the learner's module, so a wrong module shows up.
_COURSE_SLOPES = ("def _slopes(X, y, w, b):\n    r = y - (X @ w + b)\n    return -2 * (r @ X) / len(y), -2 * r.mean()\n\n\n")
_FIT_COURSE = ("import numpy as np\n\n\n" + _COURSE_SLOPES + _FIT_HEAD.split("\n\n\n", 1)[1]
               + _FIT_BODY.replace("gradients.mse_gradients(X, y, w, b)", "_slopes(X, y, w, b)"))


def _fit_course(body: str) -> str:
    """A wrong loop, with the course's own slopes."""
    return "import numpy as np\n\n\n" + _COURSE_SLOPES + _FIT_HEAD.split("\n\n\n", 1)[1] + body


_REDO = {"en": " Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.",
         "nl": " Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw."}

EXERCISES = {
    "mse-gradients": FunctionBox(
        name="mse-gradients",
        function="mse_gradients",
        start=_GRAD_START,
        solution=_GRAD,
        visible=[([[1, 2], [3, 0], [0, 1]], [3, 0, 3], [0, 0], 0)],
        hidden=[([[1], [2]], [1, 3], [1], 0), ([[1, 0, 2], [0, 1, 1], [2, 1, 0], [1, 1, 1]], [5, 2, 4, 3], [1, 1, 1], 0.5),
                ([[0.5, 1.0], [1.5, 2.0]], [1.0, 2.0], [0.2, -0.1], 1.0), (HOUSES, PRICES, [200, 20, -40], 80)],
        compare=_close_pair,
        show=_pair,
        save_as=("nn", "gradients"),
        first=[_DOTS],
        mistakes=[
            Mistake("x-at-r", syntax=r"\bX\s*@\s*r\b",
                    en="Python stops: `X @ r` puts (n, d) before (n,), and the inner numbers differ. The dot product of `r` with each column of `X` is `r @ X`.",
                    nl="Python stopt: `X @ r` zet (n, d) voor (n,), en de binnenste getallen verschillen. Het inproduct van `r` met elke kolom van `X` is `r @ X`."),
            Mistake("star", syntax=r"\br\s*\*\s*X\b",
                    en="Python stops: `r * X` multiplies place by place, and `r` and `X` do not have the same shape. A dot product with each column of `X` is `r @ X`.",
                    nl="Python stopt: `r * X` vermenigvuldigt plek voor plek, en `r` en `X` hebben niet dezelfde shape. Een inproduct met elke kolom van `X` is `r @ X`."),
            Mistake("one-value", matches=lambda got, text: not isinstance(got, (tuple, list)),
                    en="`{call}` gives `{got}`: one thing, and the check expects two, the slopes for the weights and the slope for b. Give both back: `return grad_w, grad_b`.",
                    nl="`{call}` geeft `{got}`: één ding, en de check verwacht er twee, de hellingen voor de weights en de helling voor b. Geef ze allebei terug: `return grad_w, grad_b`."),
            Mistake("sum", wrong=_grad("    r = y - (X @ w + b)\n    grad_w = -2 * (r @ X)\n    grad_b = -2 * r.sum()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add up the slopes of the examples. The MSE is a mean, so its slope is the mean of their slopes: divide by the number of examples, `len(y)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de hellingen van de voorbeelden op. De MSE is een gemiddelde, dus zijn helling is het gemiddelde van hun hellingen: deel door het aantal voorbeelden, `len(y)`."),
            Mistake("sign", wrong=_grad("    r = y - (X @ w + b)\n    grad_w = 2 * (r @ X) / len(y)\n    grad_b = 2 * r.mean()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right size, but the wrong sign. With r = y − ŷ, the inner slope of the chain rule is −x for a weight and −1 for b.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede grootte, maar het verkeerde teken. Met r = y − ŷ is de binnenste helling van de kettingregel −x voor een weight en −1 voor b."),
            Mistake("no-two", wrong=_grad("    r = y - (X @ w + b)\n    grad_w = -(r @ X) / len(y)\n    grad_b = -r.mean()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: half of each slope. The outer step is r², with the slope 2r.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de helft van elke helling. De buitenste stap is r², met helling 2r."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the residuals `r`, and the slopes `grad_w` and `grad_b` from the chain rule.",
                      "nl": "Vervang de `...` door de residuen `r`, en de hellingen `grad_w` en `grad_b` volgens de kettingregel."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, w, b)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you follow steps 1 to 3 of the recipe: the residuals, and then the mean of the chain-rule slopes for each weight and for b?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg je stap 1 tot en met 3 van het recept: de residuen, en dan het gemiddelde van de kettingregel-hellingen voor elke weight en voor b?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `X`, `y`, `w` and `b` themselves, for any number of examples and features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X`, `y`, `w` en `b` zelf, voor elk aantal voorbeelden en features?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "fit": FunctionBox(
        name="fit",
        function="fit",
        start=_FIT_START,
        solution=_FIT,
        visible=[([[1, 0], [0, 1], [1, 1]], [3, 2, 5], 0.1, 500)],
        hidden=[([[2.0]], [4.0], 0.1, 3), ([[1], [2], [3]], [3, 5, 7], 0.05, 2000),
                ([[0.5, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [2, 1, 3, 4], 0.1, 300), ([[1, 2], [2, 1]], [1, 1], 0.05, 0), ([[1, 0], [0, 1], [1, 1]], [3, 2, 5], 0.02, 100)],
        compare=_close_pair,
        show=_pair,
        save_as=("nn", "linear"),
        first=[
            _DOTS,
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: gradients_ok() is None,
                    en="Your `mse_gradients` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/gradients.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `mse_gradients` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/gradients.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-gradients", matches=lambda got, text: gradients_ok() is False,
                    en="Your loop may be right, but the `mse_gradients` in `work/projects/nn/gradients.py` no longer gives the right slopes, or stops with an error." + _REDO["en"],
                    nl="Je loop kan goed zijn, maar de `mse_gradients` in `work/projects/nn/gradients.py` geeft niet meer de goede hellingen, of stopt met een fout." + _REDO["nl"]),
        ],
        form=[Mistake("own-module", state="almost", matches=lambda got, text: re.search(r"\bmse_gradients\s*\(", text) is None,
                      en="Right, but you worked out the slopes here again. The point is to use your own function from your project: `gradients.mse_gradients(X, y, w, b)`.",
                      nl="Goed, maar je rekent de hellingen hier opnieuw uit. Het gaat erom dat je je eigen functie uit je project gebruikt: `gradients.mse_gradients(X, y, w, b)`.")],
        mistakes=[
            Mistake("bare-name", error=lambda e: isinstance(e, NameError) and "mse_gradients" in str(e),
                    en="With `{call}`, Python does not know `mse_gradients` on its own: it lives in the module you imported. Call it through the module: `gradients.mse_gradients(X, y, w, b)`.",
                    nl="Met `{call}` kent Python `mse_gradients` niet los: hij woont in de module die je binnenhaalde. Roep hem aan via de module: `gradients.mse_gradients(X, y, w, b)`."),
            Mistake("uphill", wrong=_fit_course("        grad_w, grad_b = _slopes(X, y, w, b)\n        w = w + lr * grad_w\n        b = b + lr * grad_b\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your steps go uphill, so the loss grows. An update step goes against the slope: minus.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stappen gaan bergop, dus de loss groeit. Een update-stap gaat tegen de helling in: min."),
            # Without lr the steps blow up, too far to compare in every case: the example is enough to know it.
            Mistake("no-lr", matches=lambda got, text: _close_pair(got, _NO_LR),
                    en="`{call}` gives `{got}`, and it should give `{want}`: without the learning rate, every step is the whole slope. Multiply each slope by `lr`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: zonder learning rate is elke stap de hele helling. Vermenigvuldig elke helling met `lr`."),
            Mistake("only-w", wrong=_fit_course("        grad_w, grad_b = _slopes(X, y, w, b)\n        w = w - lr * grad_w\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: b stays 0. Update b too, with its own slope: `b = b - lr * grad_b`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: b blijft 0. Doe ook een update-stap voor b, met zijn eigen helling: `b = b - lr * grad_b`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one update step: the slopes from your `gradients.mse_gradients`, and then the new `w` and `b`.",
                      "nl": "Vervang de `...` door één update-stap: de hellingen uit je `gradients.mse_gradients`, en dan de nieuwe `w` en `b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, lr, steps)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, lr, steps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each round get both slopes from `gradients.mse_gradients`, and move `w` and `b` against them, by `lr` times each slope?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Haalt elke ronde beide hellingen uit `gradients.mse_gradients`, en schuift hij `w` en `b` ertegenin, met `lr` keer elke helling?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you use `lr` and `steps` themselves?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruik je `lr` en `steps` zelf?"},
            "correct": {"en": "Your function is right in all {n} cases: for the example, `w` ends at about `[3.0, 2.0]` and `b` near 0, the weights that fit the three points exactly. The check saved it in your project, as `{saved}`. The demo below trains it on the houses.",
                        "nl": "Je functie klopt in alle {n} gevallen: in het voorbeeld eindigt `w` bij ongeveer `[3.0, 2.0]` en `b` bij 0, de weights die precies bij de drie punten passen. De check heeft hem in je project bewaard, als `{saved}`. De demo hieronder traint hem op de huizen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

def _no_lr(X, y, steps):
    """What fit gives with steps of the whole slope, without the learning rate."""
    import numpy as np
    X = np.array(X, dtype=float)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        gw, gb = _slopes(X, y, w, b)
        w, b = w - gw, b - gb
    return w, b


_NO_LR = _no_lr([[1, 0], [0, 1], [1, 1]], [3, 2, 5], 500)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
# The tests check every solution first, so mse_gradients is saved before the fit cases run.
CASES = [
    ("mse-gradients", _GRAD_START, "empty", None),
    ("mse-gradients", _GRAD, "correct", None),
    ("mse-gradients", _grad("    y_pred = X @ w + b\n    r = y - y_pred\n    n = len(y)\n    grad_w = -2 * (r @ X) / n\n    grad_b = -2 * r.sum() / n\n    return grad_w, grad_b"), "correct", None),
    ("mse-gradients", _grad("    r = (X @ w + b) - y\n    return 2 * (r @ X) / len(y), 2 * r.mean()"), "correct", None),
    ("mse-gradients", _grad("    r = y - (X @ w + b)\n    grad_w = -2 * (X @ r) / len(y)\n    grad_b = -2 * r.mean()\n    return grad_w, grad_b"), "error", "x-at-r"),
    ("mse-gradients", _grad("    r = y - (X @ w + b)\n    grad_w = -2 * (r * X).sum() / len(y)\n    grad_b = -2 * r.mean()\n    return grad_w, grad_b"), "error", "star"),
    ("mse-gradients", _grad("    r = y - (X @ w + b)\n    grad_w = -2 * (r @ X) / len(y)\n    return grad_w"), "wrong", "one-value"),
    ("mse-gradients", _grad("    r = y - (X @ w + b)\n    grad_w = -2 * (r @ X)\n    grad_b = -2 * r.sum()\n    return grad_w, grad_b"), "wrong", "sum"),
    ("mse-gradients", _grad("    r = y - (X @ w + b)\n    grad_w = 2 * (r @ X) / len(y)\n    grad_b = 2 * r.mean()\n    return grad_w, grad_b"), "wrong", "sign"),
    ("mse-gradients", _grad("    r = y - (X @ w + b)\n    grad_w = -(r @ X) / len(y)\n    grad_b = -r.mean()\n    return grad_w, grad_b"), "wrong", "no-two"),
    ("fit", _FIT_START, "empty", None),
    ("fit", _FIT, "correct", None),
    ("fit", _FIT_HEAD + "        gw, gb = gradients.mse_gradients(X, y, w, b)\n        w -= lr * gw\n        b -= lr * gb\n    return w, b", "correct", None),
    ("fit", _FIT_HEAD + "        grad_w, grad_b = mse_gradients(X, y, w, b)\n        w = w - lr * grad_w\n        b = b - lr * grad_b\n    return w, b", "error", "bare-name"),
    ("fit", _FIT_HEAD + "        grad_w, grad_b = gradients.mse_gradients(X, y, w, b)\n        w = w + lr * grad_w\n        b = b + lr * grad_b\n    return w, b", "wrong", "uphill"),
    ("fit", _FIT_HEAD + "        grad_w, grad_b = gradients.mse_gradients(X, y, w, b)\n        w = w - grad_w\n        b = b - grad_b\n    return w, b", "wrong", "no-lr"),
    ("fit", _FIT_HEAD + "        grad_w, grad_b = gradients.mse_gradients(X, y, w, b)\n        w = w - lr * grad_w\n    return w, b", "wrong", "only-w"),
    ("fit", _FIT_COURSE, "almost", "own-module"),
    ("fit", _FIT_HEAD + "        grad_w, grad_b = gradients.mse_gradients(X, y, w, b)\n        w = w - 0.1 * grad_w\n        b = b - 0.1 * grad_b\n    return w, b", "almost", None),
]
