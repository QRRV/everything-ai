# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-cross-validation: the folds of 3-fold cross-validation that choose the degree of a curve.
And the picture: for eight shuffles of the 274 penguins outside the test set, the mistakes of k-NN for every k on one
validation set of 68, and over the 5 folds of cross-validation."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # in the order np.unique sorts them, so a tie is broken the same way
SEED = 27                    # the shuffle of the lesson before, ml-knn: 206 to train, 68 to validate, 68 for the test
N_REST, N_TRAIN = 274, 206   # the penguins outside the test set, and how many of them train when one set validates
FOLDS, FOLD = 5, 55          # five folds of 55 penguins; the last one has 54
TWO = ["bill_length_mm", "body_mass_g"]
KS = [1, 3, 5, 9, 15, 25, 41, 71, 121]
SHUFFLES = list(range(1, 9))  # shuffle 1 is the order of the lesson before; the others shuffle the 274 again


@functools.lru_cache(maxsize=1)
def _data():
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(SEED).permutation(len(df))
    return np.array(df[TWO]), np.array(df["species"]), order[:N_REST], order[N_REST:]


def shuffled(s: int):
    """The positions of the 274 penguins outside the test set, in the order of shuffle s."""
    import numpy as np
    _, _, rest, _ = _data()
    return rest if s == 1 else rest[np.random.default_rng(s).permutation(N_REST)]


def _mistakes(train, val) -> list:
    """For every k in KS: how many of the penguins at the positions `val` k-NN gets wrong when it stores the penguins
    at `train`, standardized with their own mean and standard deviation. The same as asking the course's k-NN one
    penguin at a time, as the worked example does: the same distances, the same stable order, the same vote."""
    import numpy as np
    X, y, _, _ = _data()
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    A, B = (X[train] - mean) / std, (X[val] - mean) / std
    order = np.argsort(np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2), axis=1, kind="stable")
    labels = np.array([SPECIES.index(v) for v in y[train]])
    counts = np.cumsum(np.eye(len(SPECIES), dtype=int)[labels[order]], axis=1)
    truth = np.array([SPECIES.index(v) for v in y[val]])
    return [int(np.sum(np.argmax(counts[:, k - 1, :], axis=1) != truth)) for k in KS]


@functools.lru_cache(maxsize=len(SHUFFLES))
def one_split(s: int) -> tuple:
    """Mistakes per k on one validation set: the last 68 of shuffle s, with the first 206 to train on."""
    r = shuffled(s)
    return tuple(_mistakes(r[:N_TRAIN], r[N_TRAIN:]))


def folds(r):
    """The validation and training positions of each of the five folds, as the lesson cuts them."""
    import numpy as np
    return [(r[FOLD * f:FOLD * (f + 1)], np.concatenate([r[:FOLD * f], r[FOLD * (f + 1):]])) for f in range(FOLDS)]


@functools.lru_cache(maxsize=len(SHUFFLES))
def cross_validation(s: int) -> tuple:
    """Mistakes per k over the five folds of shuffle s: every one of the 274 penguins counted once."""
    total = [0] * len(KS)
    for val, train in folds(shuffled(s)):
        total = [a + b for a, b in zip(total, _mistakes(train, val))]
    return tuple(total)


def chosen(mistakes: tuple) -> int:
    """The k with the fewest mistakes; the first of them, as np.argmin takes it."""
    return KS[mistakes.index(min(mistakes))]


def test_mistakes(k: int) -> int:
    """With all 274 penguins outside the test set to vote, how many of the 68 test penguins k-NN gets wrong."""
    _, _, rest, test = _data()
    return _mistakes(rest, test)[KS.index(k)]


def spread(way: str, up_to: int = 41) -> tuple[float, float]:
    """The lowest and the highest share wrong over all eight shuffles, for k up to `up_to`: "one" for one validation
    set, "cv" for cross-validation. For the alt text of the picture."""
    curve, n = (one_split, N_REST - N_TRAIN) if way == "one" else (cross_validation, N_REST)
    shares = [m / n for s in SHUFFLES for k, m in zip(KS, curve(s)) if k <= up_to]
    return min(shares), max(shares)


def picture(s: int, labels: dict[str, str]) -> str:
    """For shuffle s: one chart the share of the 68 validation penguins k-NN gets wrong for every k, the other the share
    of all 274 over the five folds of cross-validation; k on a log scale. Thin lines: the other seven shuffles. The open
    dot: the k each way chooses. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    from everything_ai import chart
    panels = []
    for title, curve, n in ((labels["one"], one_split, N_REST - N_TRAIN), (labels["cv"], cross_validation, N_REST)):
        p = chart.Panel((0.8, 150), (0, 0.3), title=title, xlabel="k", ylabel=labels["share"], xticks=[1, 5, 15, 41, 121],
                        yticks=[0, 0.1, 0.2, 0.3], xlog=True)
        for other in SHUFFLES:
            if other != s:
                p.line(KS, [m / n for m in curve(other)], "loss", 0.8)
        mine = [m / n for m in curve(s)]
        p.line(KS, mine, "loss", 2.4)
        k = chosen(curve(s))
        p.points([k], [mine[KS.index(k)]], "loss", 6.5, hollow=True)
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "why-reliable" and "after-cv" are the choice questions in the lesson.
REQUIRED = ["why-reliable", "folds", "after-cv"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# The 18 temperatures of exercise 2 in ml-validation-set, its 12 to train and 6 to validate together.
_F_HEAD = ("import numpy as np\n\nx = np.array([8.4, 23.9, 6.0, 22.7, 5.1, 20.1, 22.4, 4.2, 9.9, 0.5, 20.8, 23.6, 10.3, 4.9, 16.0, 2.2, 16.4, 21.8])\n"
           "y = np.array([17.8, 15.2, 15.5, 15.5, 14.1, 19.4, 18.7, 15.5, 18.6, 8.8, 18.4, 17.3, 19.3, 16.2, 20.7, 12.4, 19.7, 16.3])\n"
           "order = np.random.default_rng(1).permutation(18)\ndegrees = [1, 2, 3, 4, 5, 6]\ncv_mses = []\nfor degree in degrees:\n"
           "    total = 0\n    for f in range(3):\n")
_F_TAIL = ("        w = np.polyfit(x[train], y[train], degree)\n        total = total + np.sum((y[val] - np.polyval(w, x[val])) ** 2)\n"
           "    cv_mses.append(total / 18)\ncv_mses")
_F_START = _F_HEAD + "        val = ...\n        train = ...\n" + _F_TAIL + "\n"
_F_FOLD = "        val = order[6 * f:6 * (f + 1)]\n        train = np.concatenate([order[:6 * f], order[6 * (f + 1):]])\n"
_F = _F_HEAD + _F_FOLD + _F_TAIL


def _f(fold: str) -> str:
    return _F_HEAD + fold + _F_TAIL


def _cv_mses(text: str) -> list:
    from everything_ai.check import run_code
    return run_code(text).value


_RIGHT = _cv_mses(_F)
_BEST = 1 + _RIGHT.index(min(_RIGHT))

EXERCISES = {
    "folds": CodeBox(
        name="folds",
        start=_F_START,
        solution=_F,
        digits=2,
        keep=["for degree in degrees:", "for f in range(3):"],
        # The check also runs the box with other degrees: numbers typed in by hand do not change with them.
        others=[{"degrees": Code("[1, 3, 5]")}],
        first=[_DOTS],
        mistakes=[
            # Before "one-side": with the recipe's 55, val is all 18 and train is empty, but np.concatenate was right.
            Mistake("fifty-five", syntax=r"^[^#\n]*\b55\b",
                    en="Your code stops on the line `{line}`: 55 is the size of a fold of the 274 penguins. Here 18 measurements make 3 folds of 6, so fold `f` starts at `6 * f`.",
                    nl="Je code stopt op de regel `{line}`: 55 is de grootte van een fold van de 274 pinguïns. Hier maken 18 metingen 3 folds van 6, dus fold `f` begint bij `6 * f`."),
            Mistake("no-list", error=lambda e: isinstance(e, TypeError) and "scalar index" in str(e),
                    en="Your code stops at `np.concatenate`. It takes the arrays together in one list, between square brackets: `np.concatenate([a, b])`. Without the brackets, NumPy reads the second array as something else.",
                    nl="Je code stopt bij `np.concatenate`. Die krijgt de arrays samen in één lijst, tussen vierkante haken: `np.concatenate([a, b])`. Zonder de haken leest NumPy de tweede array als iets anders."),
            Mistake("plus", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="Your code stops: with two arrays, `+` adds them element by element; it does not put them one after the other. Put the part before the fold and the part after it together with `np.concatenate([a, b])`.",
                    nl="Je code stopt: bij twee arrays telt `+` ze element voor element op; het zet ze niet achter elkaar. Zet het deel vóór de fold en het deel erna achter elkaar met `np.concatenate([a, b])`."),
            Mistake("one-side", error=lambda e: isinstance(e, TypeError) and "non-empty" in str(e),
                    en="Your code stops: for one fold, `train` is empty. The training set is everything before the fold and everything after it, put together with `np.concatenate`: for the first fold the part before is empty, for the last one the part after.",
                    nl="Je code stopt: bij één fold is `train` leeg. De training set is alles vóór de fold en alles erna, samengevoegd met `np.concatenate`: bij de eerste fold is het deel ervoor leeg, bij de laatste het deel erna."),
            Mistake("values", error=lambda e: isinstance(e, IndexError) and "integer" in str(e),
                    en="Your code stops: `val` and `train` hold positions, and you put measurements in them. Take them from `order`, the shuffled positions: `order[6 * f:6 * (f + 1)]`.",
                    nl="Je code stopt: `val` en `train` bevatten posities, en jij zet er metingen in. Haal ze uit `order`, de geschudde posities: `order[6 * f:6 * (f + 1)]`."),
            Mistake("all-train", wrong=_f("        val = order[6 * f:6 * (f + 1)]\n        train = order\n"),
                    en="Your last line gives `{got}`: low numbers, and the lowest for the highest degree, because every curve also trains on the fold it is measured on. Train on the other folds only: `np.concatenate` of the part before the fold and the part after it.",
                    nl="Je laatste regel geeft `{got}`: lage getallen, en de laagste bij de hoogste graad, omdat elke curve ook traint op de fold waarop je hem meet. Train alleen op de andere folds: `np.concatenate` van het deel vóór de fold en het deel erna."),
            Mistake("size-three", wrong=_f("        val = order[3 * f:3 * (f + 1)]\n        train = np.concatenate([order[:3 * f], order[3 * (f + 1):]])\n"),
                    en="Your last line gives `{got}`. With 3 folds of 18 measurements, each fold has 6: fold `f` starts at `6 * f`. With 3 per fold, only 9 of the 18 get a turn.",
                    nl="Je laatste regel geeft `{got}`. Met 3 folds van 18 metingen heeft elke fold er 6: fold `f` begint bij `6 * f`. Met 3 per fold komen maar 9 van de 18 aan de beurt."),
            Mistake("overlap", wrong=_f("        val = order[f:f + 6]\n        train = np.concatenate([order[:f], order[f + 6:]])\n"),
                    en="Your last line gives `{got}`. `order[f:f + 6]` lets the folds overlap: they start at 0, 1 and 2, so most measurements never get a turn and some get three. Fold `f` starts at `6 * f`.",
                    nl="Je laatste regel geeft `{got}`. Met `order[f:f + 6]` overlappen de folds: ze beginnen bij 0, 1 en 2, dus de meeste metingen komen nooit aan de beurt en sommige drie keer. Fold `f` begint bij `6 * f`."),
            Mistake("unshuffled", wrong=_f("        val = np.arange(6 * f, 6 * (f + 1))\n        train = np.concatenate([np.arange(0, 6 * f), np.arange(6 * (f + 1), 18)])\n"),
                    en="Your last line gives `{got}`: the folds of the measurements in the order of the table, not in the shuffled order. Cut the folds out of `order`.",
                    nl="Je laatste regel geeft `{got}`: de folds van de metingen in de volgorde van de tabel, niet in de geschudde volgorde. Snijd de folds uit `order`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the positions of fold `f` in `val`, and the positions of the other folds in `train`.",
                      "nl": "Vervang de twee `...`: de posities van fold `f` in `val`, en de posities van de andere folds in `train`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other degrees by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere graden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `cv_mses` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `cv_mses` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `val` the 6 positions of fold `f` in `order`, and `train` the 12 positions before and after it?",
                      "nl": "Je laatste regel geeft `{got}`. Is `val` de 6 posities van fold `f` in `order`, en `train` de 12 posities ervoor en erna?"},
            "other": {"en": "Right for these degrees. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work the folds out from `order` and `f`?",
                      "nl": "Goed voor deze graden. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je de folds uit met `order` en `f`?"},
            "other_error": {"en": "Right for these degrees. But with {where}, your code stops with an error. Do you work the folds out from `order` and `f`?",
                            "nl": "Goed voor deze graden. Maar met {where} stopt je code met een fout. Reken je de folds uit met `order` en `f`?"},
            "correct": {"en": "Every measurement had one turn in a fold. The lowest cross-validation MSE is for degree {best}, about {low}, as one validation set chose in *Choosing hyperparameters: the validation set*. The check also tried other degrees.".replace("{best}", str(_BEST)).replace("{low}", f"{min(_RIGHT):.2f}"),
                        "nl": "Elke meting was één keer aan de beurt in een fold. De laagste MSE van cross-validation hoort bij graad {best}, ongeveer {low}, zoals één validation set koos in *Hyperparameters kiezen: de validation set*. De check probeerde ook andere graden.".replace("{best}", str(_BEST)).replace("{low}", f"{min(_RIGHT):.2f}".replace(".", ","))},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("folds", _F_START, "empty", None),
    ("folds", _F, "correct", None),
    ("folds", _f("        val = order[6 * f:6 * f + 6]\n        train = np.concatenate([order[0:6 * f], order[6 * f + 6:18]])\n"), "correct", None),
    ("folds", _f("        start = 6 * f\n        end = start + 6\n        val = order[start:end]\n        train = np.concatenate([order[:start], order[end:]])\n"), "correct", None),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = order[:6 * f]\n"), "error", "one-side"),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = order[6 * (f + 1):]\n"), "error", "one-side"),
    ("folds", _f("        val = x[6 * f:6 * (f + 1)]\n        train = np.concatenate([x[:6 * f], x[6 * (f + 1):]])\n"), "error", "values"),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = order\n"), "wrong", "all-train"),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = ...\n"), "wrong", "dots-left"),
    ("folds", _f("        val = order[55 * f:55 * (f + 1)]\n        train = np.concatenate([order[:55 * f], order[55 * (f + 1):]])\n"), "error", "fifty-five"),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = np.concatenate(order[:6 * f], order[6 * (f + 1):])\n"), "error", "no-list"),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = order[:6 * f] + order[6 * (f + 1):]\n"), "error", "plus"),
    ("folds", _f("        val = order[6 * f:6 * (f + 1)]\n        train = list(order[:6 * f]) + list(order[6 * (f + 1):])\n"), "correct", None),
    ("folds", _f("        val = order[3 * f:3 * (f + 1)]\n        train = np.concatenate([order[:3 * f], order[3 * (f + 1):]])\n"), "wrong", "size-three"),
    ("folds", _f("        val = order[f:f + 6]\n        train = np.concatenate([order[:f], order[f + 6:]])\n"), "wrong", "overlap"),
    ("folds", _f("        val = np.arange(6 * f, 6 * (f + 1))\n        train = np.concatenate([np.arange(0, 6 * f), np.arange(6 * (f + 1), 18)])\n"), "wrong", "unshuffled"),
    ("folds", _F_HEAD + _F_FOLD + _F_TAIL.replace("cv_mses.append(total / 18)\ncv_mses", "cv_mses.append(total / 18)\ncv_mses = [7.26, 1.0, 1.28, 1.65, 1.25, 2.35]\ncv_mses"), "wrong", None),
    ("folds", _F_HEAD + _F_FOLD + _F_TAIL.replace("cv_mses.append(total / 18)\ncv_mses", "cv_mses.append(total / 18)\ncv_mses = " + repr([float(v) for v in _RIGHT]) + "\ncv_mses"), "almost", "other-value"),
]
