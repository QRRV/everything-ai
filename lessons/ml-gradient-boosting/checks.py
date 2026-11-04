# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-gradient-boosting: gradient boosting on the Palmer penguins, built from scikit-learn's
regression trees the way the lesson builds it. The body mass of a penguin from its bill length and bill depth: one
round of boosting, a prediction from the start and the trees, and why more trees can make it worse. And the picture:
the predicted against the real body mass after the first n trees, and the RMSE on the training and the validation set
for every n."""
import copy
import functools
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
N_TRAIN, N_VAL = 206, 68     # of the 342 penguins with all measurements, split with seed 0 as in ml-project-tabular
TWO = ["bill_length_mm", "bill_depth_mm"]
TARGET = "body_mass_g"
RATE = 0.1                   # the learning rate of the lesson
DEPTH = 2                    # the maximum depth of every tree
TREES = 1000                 # the most trees the picture shows
BEST = 42                    # where the validation RMSE is smallest; tests/test_ml_gradient_boosting.py checks it
SIZES = [1, 2, 3, 5, 10, 20, 30, 42, 60, 100, 200, 500, 1000]   # the stops of the slider: how many trees
TICKS = [1, 10, 100, 1000]
FOREST = 447                 # the RMSE of scikit-learn's forest of 100 trees, one random feature per question, on the validation set, as the test measures


@functools.lru_cache(maxsize=1)
def split():
    """The training and the validation penguins as arrays of bill length and bill depth, and their body mass, as in
    the worked example: seed 0, then 206 and 68 penguins. Read-only, so code that is given them cannot change them."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(0).permutation(len(df))
    train, val = df.iloc[order[:N_TRAIN]], df.iloc[order[N_TRAIN:N_TRAIN + N_VAL]]
    out = {"X_train": np.array(train[TWO]), "y_train": np.array(train[TARGET]),
           "X_val": np.array(val[TWO]), "y_val": np.array(val[TARGET])}
    for a in out.values():
        a.flags.writeable = False
    return out


def rmse(y, pred) -> float:
    """The root mean squared error: the square root of the mean squared residual."""
    import numpy as np
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


@functools.lru_cache(maxsize=4)
def boosted(rate: float = RATE, trees: int = TREES) -> dict:
    """Gradient boosting as the worked example does it: the mean as the start, then round after round a regression tree
    of depth 2 on the residuals, random_state 0, and a part `rate` of what it predicts added. The trees, the RMSE on
    the training and the validation set after every round, and the predictions after each number of trees in SIZES.
    About half a second for 1000 trees."""
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    s = split()
    start = float(np.mean(s["y_train"]))
    pred, pred_val = np.zeros(N_TRAIN) + start, np.zeros(N_VAL) + start
    out = {"start": start, "trees": [], "train": [], "val": [], "at": {}}
    for i in range(trees):
        tree = DecisionTreeRegressor(max_depth=DEPTH, random_state=0).fit(s["X_train"], s["y_train"] - pred)
        pred = pred + rate * tree.predict(s["X_train"])
        pred_val = pred_val + rate * tree.predict(s["X_val"])
        out["trees"].append(tree)
        out["train"].append(rmse(s["y_train"], pred))
        out["val"].append(rmse(s["y_val"], pred_val))
        if i + 1 in SIZES:
            out["at"][i + 1] = (pred.copy(), pred_val.copy())
    return out


def best(rate: float = RATE) -> int:
    """The number of trees with the smallest RMSE on the validation set."""
    curve = boosted(rate)["val"]
    return curve.index(min(curve)) + 1


# --- the picture ------------------------------------------------------------------------------------------------------

MASS = (2500, 6500)


def picture(n: int, labels: dict[str, str]) -> str:
    """Left: a dot for every validation penguin, across its real body mass and up the mass the first n trees predict,
    with its residual as a stroke to the diagonal, where the prediction is exactly right; the training penguins as
    small open grey dots. Right: the RMSE on the training set (grey) and on the validation set (purple) for 1 to 1000 trees,
    the forest's RMSE as a dashed line, and this n marked. Drawn as SVG text, once for every stop of the slider
    (L.explore); dragging across the right chart sets n."""
    from everything_ai import chart
    s, b = split(), boosted()
    pred, pred_val = b["at"][n]
    p = chart.Panel(MASS, MASS, title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[3000, 4000, 5000, 6000], yticks=[3000, 4000, 5000, 6000])
    p.line(MASS, MASS, "ink-3", 1.2, dashed=True)
    for real, said in zip(s["y_val"], pred_val):
        p.line([real, real], [said, real], "residual", 1.0)
    p.points(s["y_train"], pred, "ink-3", 2.0, hollow=True)
    p.points(s["y_val"], pred_val, "data", 3.4)
    e = chart.Panel((0.8, 1300), (0, 900), title=labels["every"], xlabel=labels["trees"], ylabel=labels["rmse"],
                    xticks=TICKS, yticks=[0, 200, 400, 600, 800], xlog=True)
    counts = list(range(1, TREES + 1))
    e.line([0.8, 1300], [FOREST, FOREST], "ink-3", 1.2, dashed=True)
    e.text(1.0, FOREST - 45, labels["forest"], "ink-3", anchor="start")
    e.line([n, n], [0, 900], "ink-3", 1.2)
    e.line(counts, b["train"], "ink-3", 1.8)
    e.line(counts, b["val"], "loss", 2.2)
    e.points([n], [b["train"][n - 1]], "ink-3", 4.0)
    e.points([n], [b["val"][n - 1]], "loss", 5.0)
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "too-many" is the choice question in the lesson.
REQUIRED = ["one-round", "predict", "too-many"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


# --- exercise 1: one round --------------------------------------------------------------------------------------------

_R_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.tree import DecisionTreeRegressor\n\n"
           'df = pd.read_csv("data/penguins.csv")\ndf = df[df["body_mass_g"].isna() == False]\n'
           "train = df.iloc[np.random.default_rng(0).permutation(len(df))[:206]]\n"
           'X = np.array(train[["bill_length_mm", "bill_depth_mm"]])\ny = np.array(train["body_mass_g"])\n'
           "rate = 0.1\npred = np.zeros(len(y)) + np.mean(y)\n")
_R_START = _R_HEAD + "residuals = ...\ntree = ...\npred = ...\npred\n"


def _r(residuals: str = "y - pred", tree: str = "DecisionTreeRegressor(max_depth=2).fit(X, residuals)",
       update: str = "pred + rate * tree.predict(X)") -> str:
    return _R_HEAD + f"residuals = {residuals}\ntree = {tree}\npred = {update}\npred\n"


_R = _r()


def _is_mean(got) -> bool:
    """Every prediction still the mean: nothing was added."""
    import numpy as np
    try:
        a = np.asarray(got, dtype=float)
    except Exception:
        return False
    return a.ndim == 1 and len(a) > 1 and np.allclose(a, a[0])


EXERCISES = {
    "one-round": CodeBox(
        name="one-round",
        start=_R_START,
        solution=_R,
        keep=["rate = 0.1"],
        # The check also runs the box with another learning rate, and with another start: a 0.1 or a mean typed in by
        # hand does not change with them.
        others=[{"rate": 0.5}, {"pred": Code("np.zeros(len(y)) + 4000.0")}],
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_r(residuals="pred - y"),
                    en="Your predictions moved away from the body masses: the training RMSE went up instead of down. A residual is the label minus the prediction, `y - pred`: positive where the prediction is too low, so adding it pushes the prediction up.",
                    nl="Je voorspellingen gingen van de lichaamsgewichten af: de RMSE op de training set ging omhoog in plaats van omlaag. Een residu is het label min de voorspelling, `y - pred`: positief waar de voorspelling te laag is, dus als je het optelt, gaat de voorspelling omhoog."),
            Mistake("no-rate", wrong=_r(update="pred + tree.predict(X)"),
                    en="Your code adds all of what the tree predicts. Gradient boosting adds only a part of it, the learning rate: `rate * tree.predict(X)`.",
                    nl="Je code telt alles op wat de boom voorspelt. Gradient boosting telt maar een deel ervan op, de learning rate: `rate * tree.predict(X)`."),
            Mistake("labels", wrong=_r(tree="DecisionTreeRegressor(max_depth=2).fit(X, y)"),
                    en="Your tree learns the body masses `y` themselves. In boosting each tree learns what is still wrong: the residuals.",
                    nl="Je boom leert de lichaamsgewichten `y` zelf. Bij boosting leert elke boom wat nog fout is: de residuen."),
            Mistake("replace", wrong=_r(update="rate * tree.predict(X)"),
                    en="Your `pred` is only what the tree adds: around 0 grams instead of around 4200. Add it to the prediction so far: `pred + rate * tree.predict(X)`.",
                    nl="Je `pred` is alleen wat de boom toevoegt: rond 0 gram in plaats van rond 4200. Tel het op bij de voorspelling tot nu toe: `pred + rate * tree.predict(X)`."),
            Mistake("no-tree", wrong=_r(update="pred + rate * residuals"),
                    en="Your code adds the residuals themselves, without the tree. On the training penguins that works, but for a new penguin you do not know its residual: you would need its body mass. The tree is what predicts the residuals for new penguins too: `rate * tree.predict(X)`.",
                    nl="Je code telt de residuen zelf op, zonder de boom. Op de trainingspinguïns werkt dat, maar van een nieuwe pinguïn ken je het residu niet: daarvoor moet je zijn gewicht al weten. De boom is wat de residuen ook voor nieuwe pinguïns voorspelt: `rate * tree.predict(X)`."),
            Mistake("deep", wrong=_r(tree="DecisionTreeRegressor().fit(X, residuals)"),
                    en="Your tree has no maximum depth, so it grows until it gets almost every residual of the training set exactly. Boosting uses small trees, which each learn only the broad lines: `max_depth=2`.",
                    nl="Je boom heeft geen maximale diepte, dus hij groeit tot hij bijna elk residu van de training set precies heeft. Boosting gebruikt kleine bomen, die elk alleen de grote lijn leren: `max_depth=2`."),
            Mistake("unchanged", matches=lambda got, text: _is_mean(got),
                    en="Every prediction in your `pred` is still the mean: nothing was added. Add a part of what the tree predicts: `pred + rate * tree.predict(X)`.",
                    nl="Elke voorspelling in je `pred` is nog het gemiddelde: er is niets bij gekomen. Tel een deel op van wat de boom voorspelt: `pred + rate * tree.predict(X)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: the residuals, a tree of depth 2 that learns them, and the new prediction.",
                      "nl": "Vervang de drie `...`: de residuen, een boom met diepte 2 die ze leert, en de nieuwe voorspelling."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another learning rate by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere learning rate."},
            "no_value": {"en": "The last line of the box shows nothing. Put `pred` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `pred` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. After one round, `pred` is the mean plus `rate` times what a tree of depth 2 predicts, a tree that learned the residuals `y - pred`. Which step is different in your code?",
                      "nl": "Je laatste regel geeft `{got}`. Na één ronde is `pred` het gemiddelde plus `rate` keer wat een boom met diepte 2 voorspelt, een boom die de residuen `y - pred` leerde. Welke stap is anders in je code?"},
            "other": {"en": "Right here. But with {where}, your `pred` is different from what it should be. Do your lines use `rate` and `pred`, instead of 0.1 and the mean?",
                      "nl": "Hier goed. Maar met {where} is je `pred` anders dan hij moet zijn. Gebruiken je regels `rate` en `pred`, in plaats van 0,1 en het gemiddelde?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, your code stops with an error. Do you use `rate` in your line, instead of 0.1?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt je code met een fout. Gebruik je `rate` in je regel, in plaats van 0,1?"},
            "correct": {"en": "One round of gradient boosting. The training RMSE fell from 821 to 770 grams: the tree found the heavy penguins, with long, shallow bills, and moved them a tenth of the way up. The check also tried another learning rate.",
                        "nl": "Eén ronde gradient boosting. De RMSE op de training set daalde van 821 naar 770 gram: de boom vond de zware pinguïns, met lange, lage snavels, en schoof ze een tiende van de weg omhoog. De check probeerde ook een andere learning rate."},
        },
    ),
}


# --- exercise 2: predict ----------------------------------------------------------------------------------------------

class _Named:
    """A test value that the check's messages write by its name, such as `trees`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _shown(v) -> str:
    """Predicted body masses as the lesson writes them: to one decimal, as a list."""
    import numpy as np
    try:
        a = np.asarray(v, dtype=float)
    except Exception:
        return str(v)
    if a.ndim == 0:
        return f"{float(a):.1f}"
    if a.ndim == 1:
        return "[" + ", ".join(f"{x:.1f}" for x in a) + "]"
    return str(v)


def _cases():
    """The calls of exercise 2: the lesson's start and trees on validation penguins."""
    s, b = split(), boosted()
    start, trees = b["start"], b["trees"]
    visible = [(_Named("start", start), _Named("trees", list(trees[:3])), 0.1, _Named("X_new", s["X_val"][:4].copy()))]
    hidden = [(_Named("start", start), _Named("trees42", list(trees[:BEST])), 0.1, _Named("X_val", s["X_val"].copy())),
              (_Named("4000.0", 4000.0), _Named("trees5", list(trees[5:10])), 0.5, _Named("X_some", s["X_val"][10:15].copy())),
              (_Named("start", start), _Named("one_tree", [trees[0]]), 0.1, _Named("X_new", s["X_val"][:4].copy()))]
    return visible, hidden


_P_HEAD = "import numpy as np\n\n\ndef predict(start, trees, rate, X):\n"
_P_START = _P_HEAD + "    pred = np.zeros(len(X)) + ...\n    for tree in trees:\n        pred = ...\n    return pred\n"


def _p(first: str = "start", update: str = "pred + rate * tree.predict(X)") -> str:
    return _P_HEAD + f"    pred = np.zeros(len(X)) + {first}\n    for tree in trees:\n        pred = {update}\n    return pred\n"


_P = _p()


_VISIBLE, _HIDDEN = _cases()


EXERCISES["predict"] = FunctionBox(
    name="predict",
    function="predict",
    start=_P_START,
    solution=_P,
    visible=_VISIBLE,
    hidden=_HIDDEN,
    show=_shown,
    first=[_DOTS],
    mistakes=[
        Mistake("no-start", wrong=_p(first="0"),
                en="`{call}` gives `{got}`: only what the trees add, around 0 grams. Every prediction begins at the start, the mean: `np.zeros(len(X)) + start`.",
                nl="`{call}` geeft `{got}`: alleen wat de bomen erbij doen, rond 0 gram. Elke voorspelling begint bij de start, het gemiddelde: `np.zeros(len(X)) + start`."),
        Mistake("no-rate", wrong=_p(update="pred + tree.predict(X)"),
                en="`{call}` gives `{got}`, and it should give `{want}`. Every tree was trained to count for a part `rate` of what it predicts: `rate * tree.predict(X)`.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke boom is getraind om mee te tellen voor een deel `rate` van wat hij voorspelt: `rate * tree.predict(X)`."),
        Mistake("last-tree", wrong=_p(update="start + rate * tree.predict(X)"),
                en="`{call}` gives `{got}`, and it should give `{want}`. Your loop starts again from `start` for every tree, so only the last tree counts. Add each tree to the prediction so far: `pred + ...`.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je lus begint bij elke boom opnieuw bij `start`, dus alleen de laatste boom telt. Tel elke boom op bij de voorspelling tot nu toe: `pred + ...`."),
        Mistake("replace", wrong=_p(update="rate * tree.predict(X)"),
                en="`{call}` gives `{got}`: a part of what the last tree predicts, around 0 grams. Add each tree to the prediction so far: `pred + rate * tree.predict(X)`.",
                nl="`{call}` geeft `{got}`: een deel van wat de laatste boom voorspelt, rond 0 gram. Tel elke boom op bij de voorspelling tot nu toe: `pred + rate * tree.predict(X)`."),
        Mistake("average", wrong=_p(update="pred + rate * tree.predict(X) / len(trees)"),
                en="`{call}` gives `{got}`, and it should give `{want}`. The trees of boosting are not averaged, as in a forest: each learned what the ones before it left, so all of them are added up.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De bomen van boosting worden niet gemiddeld, zoals in een forest: elke boom leerde wat de vorige overlieten, dus ze worden allemaal opgeteld."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the two `...`: where every prediction begins, and what each tree adds to it.",
                  "nl": "Vervang de twee `...`: waar elke voorspelling begint, en wat elke boom erbij doet."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(start, trees, rate, X)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(start, trees, rate, X)`."},
        "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every prediction begin at `start`, and does each tree add `rate` times what it predicts?",
                  "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Begint elke voorspelling bij `start`, en doet elke boom er `rate` keer zijn voorspelling bij?"},
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `start`, the `trees`, the `rate` and the `X` it gets?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `start`, de `trees`, de `rate` en de `X` die hij krijgt?"},
        "correct": {"en": "Your function is right in all {n} cases. With the 42 trees of the worked example it gives the predictions that are off by 415 grams on the validation penguins.",
                    "nl": "Je functie klopt in alle {n} gevallen. Met de 42 bomen uit het uitgewerkte voorbeeld geeft hij de voorspellingen die er op de pinguïns uit de validation set 415 gram naast zitten."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("one-round", _R_START, "empty", None),
    ("one-round", _R, "correct", None),
    ("one-round", _r(residuals="rate * (y - pred)", update="pred + tree.predict(X)"), "correct", None),
    ("one-round", _r(tree="DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, residuals)"), "correct", None),
    ("one-round", _R_HEAD + "residuals = y - pred\ntree = DecisionTreeRegressor(max_depth=2)\ntree.fit(X, residuals)\n"
                  "pred = pred + rate * tree.predict(X)\npred\n", "correct", None),
    ("one-round", _r(residuals="y - np.mean(y)"), "almost", "other-value"),
    ("one-round", _r(residuals="pred - y"), "wrong", "sign"),
    ("one-round", _r(update="pred + tree.predict(X)"), "wrong", "no-rate"),
    ("one-round", _r(tree="DecisionTreeRegressor(max_depth=2).fit(X, y)"), "wrong", "labels"),
    ("one-round", _r(update="rate * tree.predict(X)"), "wrong", "replace"),
    ("one-round", _r(update="pred + rate * residuals"), "wrong", "no-tree"),
    ("one-round", _r(tree="DecisionTreeRegressor().fit(X, residuals)"), "wrong", "deep"),
    ("one-round", _r(update="pred"), "wrong", "unchanged"),
    ("one-round", _r(update="pred + 0.1 * tree.predict(X)"), "almost", "other-value"),
    ("one-round", _R_HEAD + "residuals = y - pred\ntree = DecisionTreeRegressor(max_depth=2).fit(X, residuals)\n"
                  "pred = pred + rate * tree.predict(X)\nprint(pred)\n", "wrong", "print-last"),
    ("predict", _P_START, "empty", None),
    ("predict", _P, "correct", None),
    ("predict", _P_HEAD + "    pred = start\n    for tree in trees:\n        pred = pred + rate * tree.predict(X)\n    return pred\n", "correct", None),
    ("predict", _P_HEAD + "    return start + rate * sum(tree.predict(X) for tree in trees)\n", "correct", None),
    ("predict", _p(first="0"), "wrong", "no-start"),
    ("predict", _p(update="pred + tree.predict(X)"), "wrong", "no-rate"),
    ("predict", _p(update="start + rate * tree.predict(X)"), "wrong", "last-tree"),
    ("predict", _p(update="rate * tree.predict(X)"), "wrong", "replace"),
    ("predict", _p(update="pred + rate * tree.predict(X) / len(trees)"), "wrong", "average"),
    ("predict", _p(update="pred + 0.1 * tree.predict(X)"), "almost", None),
]
