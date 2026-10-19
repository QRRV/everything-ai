# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson ml-train-logistic: log_loss_gradients and fit, each saved in the learner's project
(work/projects/nn/) when right, the loss curve in the picture, and the demo that trains the learner's own logistic
regression on the students of ml-vocabulary."""
import math
import re

from everything_ai import projects
from everything_ai.check import FunctionBox, Mistake, run_code

# The students of ml-vocabulary whose result is known: hours of study and of sleep, and whether they passed.
STUDENTS = [[2, 6], [8, 7], [5, 8], [1, 5], [6, 6], [3, 8], [9, 8]]
PASSED = [0, 1, 1, 0, 1, 0, 1]
LR, STEPS = 0.1, 1000


def _slopes(X, y, w, b):
    """The course's own gradient of the log loss, so that a wrong module of the learner shows up."""
    import numpy as np
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    p = 1 / (1 + np.exp(-(X @ np.array(w, dtype=float) + b)))
    return (p - y) @ X / len(y), (p - y).mean()


def _fit(X, y, lr, steps):
    import numpy as np
    X = np.array(X, dtype=float)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        gw, gb = _slopes(X, y, w, b)
        w, b = w - lr * gw, b - lr * gb
    return w, b


def log_loss(X, y, w, b) -> float:
    import numpy as np
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    p = 1 / (1 + np.exp(-(X @ np.array(w, dtype=float) + b)))
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def curve_picture(labels: dict[str, str]) -> str:
    """The log loss of the students during 1000 steps of the course's own training, from 0.69 down."""
    from everything_ai import chart
    import numpy as np
    X = np.array(STUDENTS, dtype=float)
    w, b, losses = np.zeros(2), 0.0, []
    for _ in range(STEPS + 1):
        losses.append(log_loss(X, PASSED, w, b))
        gw, gb = _slopes(X, PASSED, w, b)
        w, b = w - LR * gw, b - LR * gb
    p = chart.Panel((0, STEPS), (0, 0.8), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line(list(range(STEPS + 1)), losses, "loss")
    return chart.figure([p], alt=labels["alt"], width=520, height=300)


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
    """A pair (array, number) as the lessons write it: ([-0.1667, 0.1667], -0.1667), rounded to 4 decimals."""
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
    """None when log_loss_gradients is not saved yet; else whether the saved one still gives the right slopes."""
    if not (projects.folder() / "nn" / "logistic_gradients.py").exists():
        return None
    for case in [([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]], [1.0, 0.0, 1.0], [0.5, -1.0], 0.25), (STUDENTS, PASSED, [1.0, 0.5], -8.0)]:
        run = run_code(f"from nn import logistic_gradients\nlogistic_gradients.log_loss_gradients{case!r}")
        if run.error is not None or run.timeout or not _close_pair(run.value, _slopes(*case)):
            return False
    return True


def trained():
    """For the demo: None when fit is not saved yet; "gradients" when its module is broken; False when the saved fit
    no longer trains right, or stops with an error; else (w, b) after the learner's own fit on the students."""
    if not (projects.folder() / "nn" / "logistic.py").exists():
        return None
    if gradients_ok() is not True:
        return "gradients"
    run = run_code(f"from nn import logistic\nlogistic.fit({STUDENTS!r}, {PASSED!r}, {LR}, {STEPS})")
    if run.error is not None or not _close_pair(run.value, _fit(STUDENTS, PASSED, LR, STEPS)):
        return False
    w, b = run.value
    return [float(v) for v in w], float(b)


def demo_picture(w: list, b: float, labels: dict[str, str]) -> str:
    """The students by hours of study and of sleep, passed filled and failed open, with the decision boundary of the
    learner's trained model as a line."""
    from everything_ai import chart
    p = chart.Panel((0, 10), (4, 10), title=labels["title"], xlabel=labels["hours"], ylabel=labels["sleep"])
    if abs(w[0]) > 1e-9:   # the line w0·hours + w1·sleep + b = 0, drawn from sleep 4 to 10
        sleeps = [4, 10]
        p.line([(-b - w[1] * s) / w[0] for s in sleeps], sleeps, "pred", 2.4)
    passed = [x for x, ok in zip(STUDENTS, PASSED) if ok]
    failed = [x for x, ok in zip(STUDENTS, PASSED) if not ok]
    p.points([h for h, s in passed], [s for h, s in passed], "data", 6)
    p.points([h for h, s in failed], [s for h, s in failed], "data", 6, hollow=True)
    return chart.figure([p], alt=labels["alt"], width=520, height=380)


def wrong_side(w: list, b: float) -> int:
    return sum((w[0] * h + w[1] * s + b > 0) != bool(ok) for (h, s), ok in zip(STUDENTS, PASSED))


# The lesson is done when these are correct. "one-step" is the choice question in the lesson.
REQUIRED = ["one-step", "log-loss-gradients", "fit-logistic"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")

_GRAD_HEAD = "import numpy as np\n\n\ndef log_loss_gradients(X, y, w, b):\n    X = np.array(X)\n    y = np.array(y)\n"
_GRAD_START = _GRAD_HEAD + "    ...\n    return grad_w, grad_b\n"
_GRAD_BODY = ("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = (diff @ X) / len(y)\n    grad_b = diff.mean()\n"
              "    return grad_w, grad_b")
_GRAD = _GRAD_HEAD + _GRAD_BODY


def _grad(new: str) -> str:
    return _GRAD_HEAD + new


_FIT_HEAD = ("import numpy as np\nfrom nn import logistic_gradients\n\n\ndef fit(X, y, lr, steps):\n    X = np.array(X)\n    y = np.array(y)\n"
             "    w = np.zeros(X.shape[1])\n    b = 0.0\n    for step in range(steps):\n")
_FIT_START = _FIT_HEAD + "        ...\n    return w, b\n"
_FIT_BODY = ("        grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)\n        w = w - lr * grad_w\n        b = b - lr * grad_b\n"
             "    return w, b")
_FIT = _FIT_HEAD + _FIT_BODY
# What the check compares with: the course's own slopes, not the learner's module, so a wrong module shows up.
_COURSE_SLOPES = ("def _slopes(X, y, w, b):\n    p = 1 / (1 + np.exp(-(X @ w + b)))\n    return (p - y) @ X / len(y), (p - y).mean()\n\n\n")
_FIT_COURSE = ("import numpy as np\n\n\n" + _COURSE_SLOPES + _FIT_HEAD.split("\n\n\n", 1)[1]
               + _FIT_BODY.replace("logistic_gradients.log_loss_gradients(X, y, w, b)", "_slopes(X, y, w, b)"))


def _fit_course(body: str) -> str:
    """A wrong loop, with the course's own slopes."""
    return "import numpy as np\n\n\n" + _COURSE_SLOPES + _FIT_HEAD.split("\n\n\n", 1)[1] + body


_REDO = {"en": " Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.",
         "nl": " Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw."}

EXERCISES = {
    "log-loss-gradients": FunctionBox(
        name="log-loss-gradients",
        function="log_loss_gradients",
        start=_GRAD_START,
        solution=_GRAD,
        visible=[([[2, 1], [1, 3], [0, 1]], [1, 0, 1], [0, 0], 0)],
        hidden=[([[1], [2]], [0, 1], [1], 0), ([[1, 0, 2], [0, 1, 1], [2, 1, 0], [1, 1, 1]], [1, 0, 1, 0], [0.5, -0.5, 0.2], 0.1),
                ([[0.5, 1.0], [1.5, 2.0]], [1.0, 0.0], [0.2, -0.1], 1.0), (STUDENTS, PASSED, [1, 0.5], -8)],
        compare=_close_pair,
        show=_pair,
        save_as=("nn", "logistic_gradients"),
        first=[_DOTS],
        mistakes=[
            Mistake("recipe-n", error=lambda e: isinstance(e, NameError) and "name 'n' is not" in str(e),
                    en="With `{call}`, Python does not know `n`: in the recipe it is the number of examples. In your function, that is `len(y)`.",
                    nl="Met `{call}` kent Python `n` niet: in het recept is dat het aantal voorbeelden. In je functie is dat `len(y)`."),
            Mistake("x-at-diff", syntax=r"\bX\s*@\s*(\(\s*p\s*-\s*y\s*\)|(diff|error)\b)",
                    en="Python stops: `X @ diff` puts (n, d) before (n,), and the inner numbers differ. The dot product of the differences with each column of `X` is `diff @ X`.",
                    nl="Python stopt: `X @ diff` zet (n, d) voor (n,), en de binnenste getallen verschillen. Het inproduct van de verschillen met elke kolom van `X` is `diff @ X`."),
            Mistake("star", syntax=r"(\b(diff|error)|\(\s*p\s*-\s*y\s*\))\s*\*\s*X\b",
                    en="Python stops: `diff * X` multiplies place by place, and `diff` and `X` do not have the same shape. A dot product with each column of `X` is `diff @ X`.",
                    nl="Python stopt: `diff * X` vermenigvuldigt plek voor plek, en `diff` en `X` hebben niet dezelfde shape. Een inproduct met elke kolom van `X` is `diff @ X`."),
            Mistake("one-value", matches=lambda got, text: not isinstance(got, (tuple, list)),
                    en="`{call}` gives `{got}`: one thing, and the check expects two, the slopes for the weights and the slope for b. Give both back: `return grad_w, grad_b`.",
                    nl="`{call}` geeft `{got}`: één ding, en de check verwacht er twee, de hellingen voor de weights en de helling voor b. Geef ze allebei terug: `return grad_w, grad_b`."),
            Mistake("sign", wrong=_grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = y - p\n    grad_w = (diff @ X) / len(y)\n    grad_b = diff.mean()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right size, the wrong sign. For log loss the gradient starts from the probability minus the label: `p - y`, not `y - p`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede grootte, het verkeerde teken. Bij log loss begint de gradient bij de kans min het label: `p - y`, niet `y - p`."),
            Mistake("sum", wrong=_grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = diff @ X\n    grad_b = diff.sum()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add up the slopes of the examples. The log loss is a mean, so its slope is the mean of their slopes: divide by `len(y)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de hellingen van de voorbeelden op. De log loss is een gemiddelde, dus zijn helling is het gemiddelde van hun hellingen: deel door `len(y)`."),
            Mistake("mse", wrong=_grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    r = y - p\n    grad_w = -2 * (r @ X) / len(y)\n    grad_b = -2 * r.mean()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: twice too big. That is how the slopes of the mean squared error look, with −2. For log loss there is no 2: `(p - y) @ X / len(y)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: twee keer te groot. Dat is de vorm van de hellingen van de mean squared error, met −2. Bij log loss staat er geen 2: `(p - y) @ X / len(y)`."),
            Mistake("no-sigmoid", wrong=_grad("    diff = (X @ w + b) - y\n    grad_w = (diff @ X) / len(y)\n    grad_b = diff.mean()\n    return grad_w, grad_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the logit minus the label, not the probability minus the label. Put the logit through the sigmoid first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de logit min het label, niet de kans min het label. Haal de logit eerst door de sigmoid."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the probabilities `p`, the differences `p - y`, and the slopes `grad_w` and `grad_b`.",
                      "nl": "Vervang de `...` door de kansen `p`, de verschillen `p - y`, en de hellingen `grad_w` en `grad_b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, w, b)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you follow steps 1 to 3 of the recipe: the probabilities, the differences `p - y`, and then the mean of the differences times `X`, and of the differences alone?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg je stap 1 tot en met 3 van het recept: de kansen, de verschillen `p - y`, en dan het gemiddelde van de verschillen keer `X`, en van de verschillen zelf?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `X`, `y`, `w` and `b` themselves, for any number of examples and features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X`, `y`, `w` en `b` zelf, voor elk aantal voorbeelden en features?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "fit-logistic": FunctionBox(
        name="fit-logistic",
        function="fit",
        start=_FIT_START,
        solution=_FIT,
        visible=[([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 0, 1, 0], 0.5, 200)],
        hidden=[([[2.0]], [1.0], 0.1, 3), ([[1], [2], [3], [4]], [0, 0, 1, 1], 0.2, 500),
                ([[0.5, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1, 1, 0, 1], 0.1, 300), ([[1, 2], [2, 1]], [1, 0], 0.05, 0),
                ([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 0, 1, 0], 0.2, 100)],
        compare=_close_pair,
        show=_pair,
        save_as=("nn", "logistic"),
        first=[
            _DOTS,
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: gradients_ok() is None,
                    en="Your `log_loss_gradients` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/logistic_gradients.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `log_loss_gradients` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/logistic_gradients.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-gradients", matches=lambda got, text: gradients_ok() is False,
                    en="Your loop may be right, but the `log_loss_gradients` in `work/projects/nn/logistic_gradients.py` no longer gives the right slopes, or stops with an error." + _REDO["en"],
                    nl="Je loop kan goed zijn, maar de `log_loss_gradients` in `work/projects/nn/logistic_gradients.py` geeft niet meer de goede hellingen, of stopt met een fout." + _REDO["nl"]),
        ],
        form=[Mistake("own-module", state="almost", matches=lambda got, text: re.search(r"\blog_loss_gradients\s*\(", text) is None,
                      en="Right, but you worked out the slopes here again. The point is to use your own function from your project: `logistic_gradients.log_loss_gradients(X, y, w, b)`.",
                      nl="Goed, maar je rekent de hellingen hier opnieuw uit. Het gaat erom dat je je eigen functie uit je project gebruikt: `logistic_gradients.log_loss_gradients(X, y, w, b)`.")],
        mistakes=[
            Mistake("bare-name", error=lambda e: isinstance(e, NameError) and "log_loss_gradients" in str(e),
                    en="With `{call}`, Python does not know `log_loss_gradients` on its own: it lives in the module you imported. Call it through the module: `logistic_gradients.log_loss_gradients(X, y, w, b)`.",
                    nl="Met `{call}` kent Python `log_loss_gradients` niet los: hij woont in de module die je binnenhaalde. Roep hem aan via de module: `logistic_gradients.log_loss_gradients(X, y, w, b)`."),
            Mistake("uphill", wrong=_fit_course("        grad_w, grad_b = _slopes(X, y, w, b)\n        w = w + lr * grad_w\n        b = b + lr * grad_b\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your steps go uphill, so the loss grows. An update step goes against the slope: minus.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stappen gaan bergop, dus de loss groeit. Een update-stap gaat tegen de helling in: min."),
            Mistake("no-lr", wrong=_fit_course("        grad_w, grad_b = _slopes(X, y, w, b)\n        w = w - grad_w\n        b = b - grad_b\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: without the learning rate, every step is the whole slope. Multiply each slope by `lr`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: zonder learning rate is elke stap de hele helling. Vermenigvuldig elke helling met `lr`."),
            Mistake("only-w", wrong=_fit_course("        grad_w, grad_b = _slopes(X, y, w, b)\n        w = w - lr * grad_w\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: b stays 0. Update b too, with its own slope: `b = b - lr * grad_b`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: b blijft 0. Doe ook een update-stap voor b, met zijn eigen helling: `b = b - lr * grad_b`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one update step: the slopes from your `logistic_gradients.log_loss_gradients`, and then the new `w` and `b`.",
                      "nl": "Vervang de `...` door één update-stap: de hellingen uit je `logistic_gradients.log_loss_gradients`, en dan de nieuwe `w` en `b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, lr, steps)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, lr, steps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each round get both slopes from `logistic_gradients.log_loss_gradients`, and move `w` and `b` against them, by `lr` times each slope?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Haalt elke ronde beide hellingen uit `logistic_gradients.log_loss_gradients`, en schuift hij `w` en `b` ertegenin, met `lr` keer elke helling?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you use `lr` and `steps` themselves?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruik je `lr` en `steps` zelf?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example, the first weight grows to about 5.9 and the second stays small, about −0.4: only the first feature tells the classes apart. The check saved it in your project, as `{saved}`. The demo below trains it on the students.",
                        "nl": "Je functie klopt in alle {n} gevallen: in het voorbeeld groeit de eerste weight tot ongeveer 5,9 en blijft de tweede bij −0,4: alleen de eerste feature houdt de klassen uit elkaar. De check heeft hem in je project bewaard, als `{saved}`. De demo hieronder traint hem op de leerlingen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
# The tests check every solution first, so log_loss_gradients is saved before the fit cases run.
CASES = [
    ("log-loss-gradients", _GRAD_START, "empty", None),
    ("log-loss-gradients", _GRAD, "correct", None),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    return (p - y) @ X / len(y), (p - y).mean()"), "correct", None),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = (X @ diff) / len(y)\n    grad_b = diff.mean()\n    return grad_w, grad_b"), "error", "x-at-diff"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    return X @ (p - y) / len(y), (p - y).mean()"), "error", "x-at-diff"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = (diff * X).sum() / len(y)\n    grad_b = diff.mean()\n    return grad_w, grad_b"), "error", "star"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = (diff @ X) / len(y)\n    return grad_w"), "wrong", "one-value"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = y - p\n    grad_w = (diff @ X) / len(y)\n    grad_b = diff.mean()\n    return grad_w, grad_b"), "wrong", "sign"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = diff @ X\n    grad_b = diff.sum()\n    return grad_w, grad_b"), "wrong", "sum"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    r = y - p\n    grad_w = -2 * (r @ X) / len(y)\n    grad_b = -2 * r.mean()\n    return grad_w, grad_b"), "wrong", "mse"),
    ("log-loss-gradients", _grad("    p = 1 / (1 + np.exp(-(X @ w + b)))\n    diff = p - y\n    grad_w = (diff @ X) / n\n    grad_b = diff.mean()\n    return grad_w, grad_b"), "error", "recipe-n"),
    ("log-loss-gradients", _grad("    diff = (X @ w + b) - y\n    grad_w = (diff @ X) / len(y)\n    grad_b = diff.mean()\n    return grad_w, grad_b"), "wrong", "no-sigmoid"),
    ("fit-logistic", _FIT_START, "empty", None),
    ("fit-logistic", _FIT, "correct", None),
    ("fit-logistic", _FIT_HEAD + "        gw, gb = logistic_gradients.log_loss_gradients(X, y, w, b)\n        w -= lr * gw\n        b -= lr * gb\n    return w, b", "correct", None),
    ("fit-logistic", _FIT_HEAD + "        grad_w, grad_b = log_loss_gradients(X, y, w, b)\n        w = w - lr * grad_w\n        b = b - lr * grad_b\n    return w, b", "error", "bare-name"),
    ("fit-logistic", _FIT_HEAD + "        grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)\n        w = w + lr * grad_w\n        b = b + lr * grad_b\n    return w, b", "wrong", "uphill"),
    ("fit-logistic", _FIT_HEAD + "        grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)\n        w = w - grad_w\n        b = b - grad_b\n    return w, b", "wrong", "no-lr"),
    ("fit-logistic", _FIT_HEAD + "        grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)\n        w = w - lr * grad_w\n    return w, b", "wrong", "only-w"),
    ("fit-logistic", _FIT_COURSE, "almost", "own-module"),
    ("fit-logistic", _FIT_HEAD + "        grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)\n        w = w - 0.5 * grad_w\n        b = b - 0.5 * grad_b\n    return w, b", "almost", None),
]
