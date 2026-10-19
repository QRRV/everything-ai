# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-train-test-split: shuffling ten houses and splitting them into a training set and a test
set, and the mean squared error of a line on both. And the picture of the forty houses of the worked example, with a
line and a look-up model trained on one shuffle and measured on the houses they did not see."""
import re

from everything_ai.check import Code, CodeBox, Mistake

SEED, N, N_TRAIN = 18, 40, 32   # as in the worked example


def _houses():
    """The forty made-up houses of the worked example: area in hundreds of m², price in thousands of euros. The
    generator is returned too, so that each next shuffle asks the same generator again, as the example does."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    sizes = 0.5 + 1.5 * rng.random(N)
    prices = 60 + 150 * sizes + 80 * (rng.random(N) - 0.5)
    return rng, sizes, prices


def _shuffles(n: int):
    """The positions of the first `n` shuffles, each split into training and test positions."""
    rng, sizes, prices = _houses()
    out = []
    for _ in range(n):
        order = rng.permutation(N)
        out.append((order[:N_TRAIN], order[N_TRAIN:]))
    return sizes, prices, out


SIZES, PRICES, SPLITS = _shuffles(8)


def _line(x, y):
    """The line with the lowest mean squared error, the one gradient descent finds: w is the mean of the products of
    the deviations from the means, divided by the mean squared deviation of x."""
    w = ((x - x.mean()) * (y - y.mean())).mean() / ((x - x.mean()) ** 2).mean()
    return w, y.mean() - w * x.mean()


def _look_up(x_train, y_train, x):
    """For each area, the price of the training house with the closest area."""
    import numpy as np
    return np.array([y_train[np.abs(x_train - v).argmin()] for v in x])


def errors(k: int) -> dict:
    """For shuffle k (1 to 8): the MSE of the line and of the look-up model, on the training set and on the test set."""
    import numpy as np
    train, test = SPLITS[k - 1]
    w, b = _line(SIZES[train], PRICES[train])
    mse = lambda y, p: float(np.mean((y - p) ** 2))
    return {"line_train": mse(PRICES[train], w * SIZES[train] + b), "line_test": mse(PRICES[test], w * SIZES[test] + b),
            "look_train": 0.0, "look_test": mse(PRICES[test], _look_up(SIZES[train], PRICES[train], SIZES[test]))}


def picture(k: int, labels: dict[str, str]) -> str:
    """Two panels with the forty houses of shuffle k, the training houses filled and the test houses open: one with
    the line, one with the look-up model, each trained on the training houses only. Drawn as SVG text, once for every
    stop of the slider (L.explore)."""
    import numpy as np
    from everything_ai import chart
    train, test = SPLITS[k - 1]
    w, b = _line(SIZES[train], PRICES[train])
    panels = [chart.Panel((0.4, 2.1), (100, 400), title=title, xlabel=labels["x"], ylabel=labels["y"])
              for title in (labels["line"], labels["look"])]
    panels[0].line([0.45, 2.05], [w * 0.45 + b, w * 2.05 + b], "pred", 2.4)
    # the look-up model is a staircase: each training house rules up to halfway to the next
    xs = np.sort(SIZES[train])
    ys = _look_up(SIZES[train], PRICES[train], xs)
    edges = [0.45] + [(a + c) / 2 for a, c in zip(xs[:-1], xs[1:])] + [2.05]
    px, py = [], []
    for i, y in enumerate(ys):
        px += [edges[i], edges[i + 1]]
        py += [y, y]
    panels[1].line(px, py, "pred", 2.0)
    for c in panels:   # the dots on top of the lines: the staircase runs through every training house
        c.points(SIZES[train], PRICES[train], "data", 4)
        c.points(SIZES[test], PRICES[test], "data", 4.5, hollow=True)
    return chart.figure(panels, alt=labels["alt"], width=330, height=280)


# The lesson is done when these are correct. "sorted" is the choice question in the lesson.
REQUIRED = ["sorted", "split", "two-mse"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_S_HEAD = ("import numpy as np\n\nsizes = np.array([0.6, 1.4, 0.9, 1.8, 1.1, 0.7, 1.5, 1.2, 0.5, 1.9])\n"
           "prices = np.array([150, 270, 200, 330, 230, 160, 290, 240, 130, 350])\nrng = np.random.default_rng(1)\n")
_S_TAIL = "x_train = sizes[train]\nx_test = sizes[test]\nx_train, x_test"
_S_START = _S_HEAD + "order = ...\ntrain = ...\ntest = ...\n" + _S_TAIL + "\n"
_S_BODY = "order = rng.permutation(10)\ntrain = order[:8]\ntest = order[8:]\n"
_S = _S_HEAD + _S_BODY + _S_TAIL


def _s(body: str) -> str:
    return _S_HEAD + body + _S_TAIL


def _lengths(got, n_train: int, n_test: int) -> bool:
    return isinstance(got, tuple) and len(got) == 2 and all(hasattr(v, "__len__") for v in got) \
        and len(got[0]) == n_train and len(got[1]) == n_test


_M_HEAD = ("import numpy as np\n\nx_train = np.array([0.5, 1.1, 1.2, 0.6, 1.4, 0.9, 0.7, 1.9])\n"
           "y_train = np.array([130, 230, 240, 150, 270, 200, 160, 350])\nx_test = np.array([1.5, 1.8])\n"
           "y_test = np.array([290, 330])\nw = 160\nb = 50\n")
_M_START = _M_HEAD + "mse_train = ...\nmse_test = ...\nmse_train, mse_test\n"
_M_BODY = "mse_train = np.mean((y_train - (w * x_train + b)) ** 2)\nmse_test = np.mean((y_test - (w * x_test + b)) ** 2)\n"
_M = _M_HEAD + _M_BODY + "mse_train, mse_test"


def _m(body: str) -> str:
    return _M_HEAD + body + "mse_train, mse_test"


EXERCISES = {
    "split": CodeBox(
        name="split",
        start=_S_START,
        solution=_S,
        keep=["rng = np.random.default_rng(1)", "x_train = sizes[train]", "x_test = sizes[test]"],
        # The check also runs the box with other areas: a split typed in by hand does not change with them.
        others=[{"sizes": Code("np.array([0.8, 1.0, 1.3, 0.4, 1.6, 2.0, 0.9, 1.7, 1.2, 0.5])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-shuffle", wrong=_s("order = np.arange(10)\ntrain = order[:8]\ntest = order[8:]\n"),
                    en="Your training set is the first 8 houses and your test set the last 2, in the order they come. Shuffle the positions first: `rng.permutation(10)`.",
                    nl="Je training set is de eerste 8 huizen en je test set de laatste 2, in de volgorde waarin ze staan. Schud eerst de posities: `rng.permutation(10)`."),
            Mistake("swapped", matches=lambda got, text: _lengths(got, 2, 8),
                    en="Your training set has 2 houses and your test set 8: the other way round. The first 8 positions are for training: `order[:8]`, and the last 2 for testing: `order[8:]`.",
                    nl="Je training set heeft 2 huizen en je test set 8: andersom. De eerste 8 posities zijn om te trainen: `order[:8]`, en de laatste 2 om te testen: `order[8:]`."),
            Mistake("all-train", matches=lambda got, text: _lengths(got, 10, 2),
                    en="Your training set has all 10 houses, so the 2 test houses are in it too: the model would see them while it learns. Take only the first 8 positions: `order[:8]`.",
                    nl="Je training set heeft alle 10 huizen, dus de 2 huizen van de test set zitten er ook in: het model zou ze zien terwijl het leert. Neem alleen de eerste 8 posities: `order[:8]`."),
            Mistake("overlap", matches=lambda got, text: isinstance(got, tuple) and len(got) == 2 and all(hasattr(v, "__len__") for v in got)
                    and len(got[0]) + len(got[1]) != 10,
                    en="Your training set and your test set do not add up to the 10 houses: a house is in both, or in neither. Split the shuffled positions in one place: `order[:8]` and `order[8:]`.",
                    nl="Je training set en je test set zijn samen niet de 10 huizen: een huis zit in allebei, of in geen van beide. Splits de geschudde posities op één plek: `order[:8]` en `order[8:]`."),
            Mistake("shuffled-values", syntax=r"permutation\(\s*(sizes|prices)\b",
                    en="On the line `{line}` you shuffle the values themselves, so `order` holds areas or prices instead of positions, and `sizes[train]` cannot select by them. Shuffle the positions: `rng.permutation(10)`.",
                    nl="Op de regel `{line}` schud je de waarden zelf, dus staan er oppervlaktes of prijzen in `order` in plaats van posities, en daarmee kan `sizes[train]` niet kiezen. Schud de posities: `rng.permutation(10)`."),
            Mistake("two-shuffles", wrong=_s("order = rng.permutation(10)\ntrain = rng.permutation(10)[:8]\ntest = rng.permutation(10)[8:]\n"),
                    en="You shuffle again for each set, so the two sets come from different orders: a house can end up in both, or in neither. Shuffle once, into `order`, and split that.",
                    nl="Je schudt voor elke set opnieuw, dus de twee sets komen uit verschillende volgordes: een huis kan in allebei terechtkomen, of in geen van beide. Schud één keer, in `order`, en splits dat."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: the shuffled positions in `order`, the first 8 of them in `train`, and the last 2 in `test`.",
                      "nl": "Vervang de drie `...`: de geschudde posities in `order`, de eerste 8 daarvan in `train`, en de laatste 2 in `test`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other houses by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere huizen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `x_train, x_test` on the last line, so the check can see them.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `x_train, x_test` op de laatste regel, zodat de check ze kan zien."},
            "wrong": {"en": "Your last line gives other houses than the check expects. Do you shuffle the positions once with `rng.permutation(10)`, and take the first 8 for training and the last 2 for testing?",
                      "nl": "Je laatste regel geeft andere huizen dan de check verwacht. Schud je de posities één keer met `rng.permutation(10)`, en neem je de eerste 8 om te trainen en de laatste 2 om te testen?"},
            "other": {"en": "Right for these houses. But with {where}, your last line gives other houses than it should. Do you select them from `sizes` by the positions in `train` and `test`?",
                      "nl": "Goed voor deze huizen. Maar met {where} geeft je laatste regel andere huizen dan zou moeten. Kies je ze uit `sizes` met de posities in `train` en `test`?"},
            "other_error": {"en": "Right for these houses. But with {where}, your code stops with an error. Do you select them from `sizes` by the positions in `train` and `test`?",
                            "nl": "Goed voor deze huizen. Maar met {where} stopt je code met een fout. Kies je ze uit `sizes` met de posities in `train` en `test`?"},
            "correct": {"en": "8 houses to train on and 2 that the model will not see until the end, the houses of 1.5 and 1.8. The check also tried other houses.",
                        "nl": "8 huizen om op te trainen en 2 die het model pas aan het eind ziet, de huizen van 1,5 en 1,8. De check probeerde ook andere huizen."},
        },
    ),
    "two-mse": CodeBox(
        name="two-mse",
        start=_M_START,
        solution=_M,
        keep=["w = 160", "b = 50"],
        # The check also runs the box with another line: numbers typed in by hand do not change with it.
        others=[{"w": Code("140")}],
        first=[_DOTS],
        mistakes=[
            Mistake("both-train", wrong=_m("mse_train = np.mean((y_train - (w * x_train + b)) ** 2)\nmse_test = np.mean((y_train - (w * x_train + b)) ** 2)\n"),
                    en="Your last line gives the same number twice: the MSE on the training houses. For the test MSE, use `x_test` and `y_test`.",
                    nl="Je laatste regel geeft twee keer hetzelfde getal: de MSE op de training set. Gebruik voor de test-MSE `x_test` en `y_test`."),
            Mistake("swapped", wrong=_m("mse_train = np.mean((y_test - (w * x_test + b)) ** 2)\nmse_test = np.mean((y_train - (w * x_train + b)) ** 2)\n"),
                    en="Your last line gives `{got}`: the two the other way round. `mse_train` belongs to `x_train` and `y_train`, `mse_test` to `x_test` and `y_test`.",
                    nl="Je laatste regel geeft `{got}`: de twee andersom. `mse_train` hoort bij `x_train` en `y_train`, `mse_test` bij `x_test` en `y_test`."),
            Mistake("no-square", wrong=_m("mse_train = np.mean(y_train - (w * x_train + b))\nmse_test = np.mean(y_test - (w * x_test + b))\n"),
                    en="Your last line gives `{got}`: the mean residuals, without squaring them. Positive and negative residuals then cancel. Square each residual first: `** 2`.",
                    nl="Je laatste regel geeft `{got}`: de gemiddelde residuen, zonder ze te kwadrateren. Positieve en negatieve residuen vallen dan tegen elkaar weg. Kwadrateer eerst elk residu: `** 2`."),
            Mistake("sum", wrong=_m("mse_train = np.sum((y_train - (w * x_train + b)) ** 2)\nmse_test = np.sum((y_test - (w * x_test + b)) ** 2)\n"),
                    en="Your last line gives `{got}`: the sums of the squared residuals. The MSE is their mean, so that a set of 2 houses and a set of 8 can be compared: `np.mean`.",
                    nl="Je laatste regel geeft `{got}`: de sommen van de gekwadrateerde residuen. De MSE is hun gemiddelde, zodat je een set van 2 huizen met een set van 8 kunt vergelijken: `np.mean`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the mean squared error of the line on the training houses, and on the test houses.",
                      "nl": "Vervang de twee `...`: de mean squared error van de lijn op de training set, en op de test set."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another line by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere lijn."},
            "no_value": {"en": "The last line of the box shows nothing. Put `mse_train, mse_test` on the last line, so the check can see them.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `mse_train, mse_test` op de laatste regel, zodat de check ze kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you take, for each set, the mean of the squared residuals `y - (w * x + b)`, with the `x` and `y` of that set?",
                      "nl": "Je laatste regel geeft `{got}`. Neem je voor elke set het gemiddelde van de gekwadrateerde residuen `y - (w * x + b)`, met de `x` en `y` van die set?"},
            "other": {"en": "Right for this line. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `w` and `b` themselves?",
                      "nl": "Goed voor deze lijn. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `w` en `b` zelf?"},
            "other_error": {"en": "Right for this line. But with {where}, your code stops with an error. Do you work it out with `w` and `b` themselves?",
                            "nl": "Goed voor deze lijn. Maar met {where} stopt je code met een fout. Reken je het uit met `w` en `b` zelf?"},
            "correct": {"en": "13.5 on the training houses and 32 on the test houses: on houses that were not in the training set, the line is further off. The check also tried another line.",
                        "nl": "13,5 op de training set en 32 op de test set: op huizen die niet in de training set zaten, zit de lijn er verder naast. De check probeerde ook een andere lijn."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("split", _S_START, "empty", None),
    ("split", _S, "correct", None),
    ("split", _s("order = rng.permutation(len(sizes))\ntrain = order[:8]\ntest = order[8:]\n"), "correct", None),
    ("split", _s("order = rng.permutation(10)\ntrain = order[:-2]\ntest = order[-2:]\n"), "correct", None),
    ("split", _s("order = rng.permutation(np.arange(10))\ntrain = order[0:8]\ntest = order[8:10]\n"), "correct", None),
    ("split", _s("order = np.arange(10)\ntrain = order[:8]\ntest = order[8:]\n"), "wrong", "no-shuffle"),
    ("split", _s("order = rng.permutation(10)\ntrain = order[8:]\ntest = order[:8]\n"), "wrong", "swapped"),
    ("split", _s("order = rng.permutation(10)\ntrain = order\ntest = order[8:]\n"), "wrong", "all-train"),
    ("split", _s("order = rng.permutation(10)\ntrain = order[:8]\ntest = order[7:]\n"), "wrong", "overlap"),
    ("split", _s("order = rng.permutation(10)\ntrain = rng.permutation(10)[:8]\ntest = rng.permutation(10)[8:]\n"), "wrong", "two-shuffles"),
    ("split", _s("order = rng.permutation(sizes)\ntrain = order[:8]\ntest = order[8:]\n"), "error", "shuffled-values"),
    ("split", _s("order = np.arange(10)\nrng.shuffle(order)\ntrain = order[:8]\ntest = order[8:]\n"), "correct", None),
    ("split", _s("order = rng.permutation(10)\ntrain = np.array([8, 4, 7, 0, 1, 2, 5, 9])\ntest = np.array([6, 3])\n"), "correct", None),   # the positions do not depend on the areas
    ("two-mse", _M_START, "empty", None),
    ("two-mse", _M, "correct", None),
    ("two-mse", _m("mse_train = ((y_train - w * x_train - b) ** 2).mean()\nmse_test = ((y_test - w * x_test - b) ** 2).mean()\n"), "correct", None),
    ("two-mse", _m("mse_train = np.mean((w * x_train + b - y_train) ** 2)\nmse_test = np.sum((w * x_test + b - y_test) ** 2) / len(y_test)\n"), "correct", None),
    ("two-mse", _m("mse_train = np.mean((y_train - (w * x_train + b)) ** 2)\nmse_test = np.mean((y_train - (w * x_train + b)) ** 2)\n"), "wrong", "both-train"),
    ("two-mse", _m("mse_train = np.mean((y_test - (w * x_test + b)) ** 2)\nmse_test = np.mean((y_train - (w * x_train + b)) ** 2)\n"), "wrong", "swapped"),
    ("two-mse", _m("mse_train = np.mean(y_train - (w * x_train + b))\nmse_test = np.mean(y_test - (w * x_test + b))\n"), "wrong", "no-square"),
    ("two-mse", _m("mse_train = np.sum((y_train - (w * x_train + b)) ** 2)\nmse_test = np.sum((y_test - (w * x_test + b)) ** 2)\n"), "wrong", "sum"),
    ("two-mse", _m("mse_train = 13.5\nmse_test = 32.0\n"), "almost", "other-value"),
    ("two-mse", _m("mse_train = np.mean(np.abs(y_train - (w * x_train + b)))\nmse_test = np.mean(np.abs(y_test - (w * x_test + b)))\n"), "wrong", None),
]
