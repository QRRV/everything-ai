# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-project-fair-comparison: the models of the module More models side by side on one question,
the sex of a Palmer penguin from its four measurements. Every model in its own pipeline, on the same five folds, next
to a baseline; two models fold by fold; k chosen with cross-validation and the test set used once; and what the test
set may and may not decide. And the picture: the score of every model on every fold, for nine ways of cutting the same
penguins into five folds, and the mean of each model for all nine."""
import functools
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import yours

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
N_COMPARE = 266              # of the 333 penguins with a known sex, shuffled with seed 0: the rest, 67, is the test set
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET = "sex"
MODELS = ["logistic", "knn", "tree", "forest", "boosting"]   # the rows of the picture, top to bottom; and "baseline"
KS = [1, 3, 5, 7, 9, 11, 15, 21, 31]                         # the values of k that exercise 2 tries
CUTS = list(range(1, 10))   # the stops of the slider: 1 is KFold(5), 2 to 9 are KFold(5, shuffle=True, random_state=0 to 7)


def cv(cut: int = 1):
    """One way of cutting the penguins into 5 folds: in order, as in the lesson, or shuffled with a seed."""
    from sklearn.model_selection import KFold
    return KFold(5) if cut == 1 else KFold(5, shuffle=True, random_state=cut - 2)


def model(name: str):
    """A fresh model of the lesson: standardizing first where the model needs it, and its settings."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    return {"baseline": lambda: DummyClassifier(),
            "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression()),
            "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
            "tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
            "forest": lambda: RandomForestClassifier(random_state=0),
            "boosting": lambda: HistGradientBoostingClassifier(random_state=0)}[name]()


@functools.lru_cache(maxsize=1)
def split():
    """The penguins to compare on and the test penguins, as arrays of the four measurements, and their sex, as in the
    worked example: the 333 with a known sex, shuffled with seed 0, then 266 and 67. Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["sex"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(0).permutation(len(df))
    compare, test = df.iloc[order[:N_COMPARE]], df.iloc[order[N_COMPARE:]]
    out = {"X": np.array(compare[FOUR]), "y": np.array(compare[TARGET]),
           "X_test": np.array(test[FOUR]), "y_test": np.array(test[TARGET])}
    for a in out.values():
        a.flags.writeable = False
    return out


def scores(name: str, cut: int = 1) -> list:
    """The accuracy on each of the 5 folds, rounded to 4 decimals. The forest and boosting take seconds for all nine
    cuts, so the picture uses MEASURED, which tests/test_ml_project_fair_comparison.py compares with this."""
    from sklearn.model_selection import cross_val_score
    s = split()
    return [round(float(v), 4) for v in cross_val_score(model(name), s["X"], s["y"], cv=cv(cut))]


def mean(values) -> float:
    return sum(values) / len(values)


SIZES = [54, 53, 53, 53, 53]   # the penguins in each of the 5 folds of 266, for every cut


def right(name: str, cut: int) -> int:
    """How many of the 266 penguins the model gets right on the fold they are in, for this cut, from MEASURED."""
    return sum(round(a * n) for a, n in zip(MEASURED[name][cut - 1], SIZES))


# scores(name, cut) for every model and every cut of the slider, cuts 1 to 9 in that order.
MEASURED = {
    "baseline": [
        [0.5926, 0.4717, 0.434, 0.434, 0.3208],
        [0.463, 0.5472, 0.5283, 0.5094, 0.566],
        [0.5741, 0.4906, 0.3774, 0.434, 0.4906],
        [0.4815, 0.5472, 0.5094, 0.5094, 0.566],
        [0.4815, 0.5472, 0.5849, 0.566, 0.434],
        [0.537, 0.6038, 0.4717, 0.4906, 0.5094],
        [0.5556, 0.5472, 0.4717, 0.5283, 0.5094],
        [0.5556, 0.4906, 0.5283, 0.5283, 0.5094],
        [0.537, 0.4717, 0.4906, 0.566, 0.5472]],
    "logistic": [
        [0.963, 0.8491, 0.9434, 0.8679, 0.9057],
        [0.9074, 0.9057, 0.9245, 0.9245, 0.9057],
        [0.963, 0.8491, 0.8868, 0.8679, 0.9245],
        [0.9074, 0.8868, 0.9811, 0.9434, 0.8302],
        [0.9074, 0.9057, 0.9245, 0.8679, 0.9057],
        [0.9259, 0.9434, 0.8491, 0.8302, 0.9245],
        [0.9259, 0.8868, 0.9057, 0.8679, 0.9434],
        [0.9074, 0.9057, 0.9245, 0.8302, 0.9245],
        [1, 1, 0.7925, 0.8868, 0.8868]],
    "knn": [
        [0.9815, 0.8868, 0.9623, 0.8302, 0.9057],
        [0.9444, 0.9245, 0.9245, 0.8868, 0.8868],
        [0.963, 0.9057, 0.8491, 0.9245, 0.9057],
        [0.963, 0.9245, 0.9434, 0.9434, 0.8491],
        [0.8519, 0.9623, 0.9434, 0.8868, 0.9245],
        [0.9259, 0.9811, 0.9057, 0.8302, 0.9623],
        [0.8889, 0.9434, 0.8679, 0.9811, 0.9057],
        [0.9444, 0.8679, 0.9434, 0.9434, 0.9057],
        [0.963, 0.9811, 0.8868, 0.8868, 0.8679]],
    "tree": [
        [0.9444, 0.7736, 0.9057, 0.8302, 0.8679],
        [0.8704, 0.8679, 0.9245, 0.8679, 0.9057],
        [0.963, 0.8491, 0.8302, 0.8679, 0.8868],
        [0.8889, 0.8868, 0.8868, 0.8679, 0.9245],
        [0.8704, 0.8679, 0.7736, 0.9057, 0.9434],
        [0.8333, 0.7925, 0.8491, 0.7547, 0.9245],
        [0.9074, 0.9057, 0.9057, 0.8868, 0.8679],
        [0.8889, 0.8679, 0.9245, 0.8491, 0.9057],
        [0.9259, 0.8679, 0.8113, 0.9057, 0.9245]],
    "forest": [
        [0.9815, 0.8679, 0.9057, 0.8679, 0.8868],
        [0.9074, 0.9623, 0.8679, 0.9245, 0.9057],
        [0.963, 0.9245, 0.8679, 0.8679, 0.9245],
        [0.8889, 0.8868, 0.9434, 0.9057, 0.8679],
        [0.8519, 0.8868, 0.9245, 0.8868, 0.9623],
        [0.9259, 0.9623, 0.8491, 0.8679, 0.9434],
        [0.8704, 0.9245, 0.8868, 0.8868, 0.9245],
        [0.9259, 0.9245, 0.9434, 0.8491, 0.9245],
        [0.9444, 0.9434, 0.8491, 0.9057, 0.9057]],
    "boosting": [
        [0.9444, 0.8868, 0.9057, 0.8679, 0.8679],
        [0.9259, 0.9057, 0.8302, 0.9057, 0.9057],
        [0.9815, 0.9245, 0.8491, 0.8491, 0.9057],
        [0.8519, 0.8868, 0.9245, 0.9245, 0.8302],
        [0.8704, 0.8868, 0.9057, 0.8679, 0.9811],
        [0.9259, 0.9623, 0.8302, 0.8679, 0.9057],
        [0.8333, 0.9057, 0.8868, 0.9245, 0.9057],
        [0.9444, 0.8302, 0.9057, 0.8302, 0.9057],
        [0.8704, 0.9434, 0.8113, 0.8868, 0.8868]],
}


# --- the picture ------------------------------------------------------------------------------------------------------

def picture(cut: int, labels: dict[str, str]) -> str:
    """Left: a row for every model, with an open dot for its accuracy on each of the 5 folds of this cut and a filled
    dot for their mean. Right: the same rows, with an open dot for the mean of each of the nine cuts and a filled dot
    for this one. The baseline, around 0.5, is left out, so that the models can be told apart; the text names it.
    Drawn as SVG text, once for every stop of the slider (L.explore)."""
    from everything_ai import chart
    rows = {name: len(MODELS) - i for i, name in enumerate(MODELS)}
    p = chart.Panel((0.74, 1.01), (0.4, len(MODELS) + 0.8), title=labels["title"], xlabel=labels["score"],
                    xticks=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0], yticks=[])
    e = chart.Panel((0.82, 0.94), (0.4, len(MODELS) + 0.8), title=labels["every"], xlabel=labels["mean"],
                    xticks=[0.82, 0.86, 0.9, 0.94], yticks=[])
    for name, row in rows.items():
        folds = MEASURED[name][cut - 1]
        p.points(folds, [row] * len(folds), "ink-3", 3.4, hollow=True)
        p.points([mean(folds)], [row], "ink", 5.0)
        p.text(0.745, row + 0.4, labels[name], "ink-2", anchor="start")
        means = [mean(f) for f in MEASURED[name]]
        e.points(means, [row] * len(means), "ink-3", 3.4, hollow=True)
        e.points([means[cut - 1]], [row], "ink", 5.0)
        e.text(0.822, row + 0.4, labels[name], "ink-2", anchor="start")
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300)


# The lesson is done when these are correct. "test-pick" is the choice question in the lesson; the written report has
# no check, as in ml-project-tabular.
REQUIRED = ["fold-by-fold", "choose-k", "test-pick"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
         "from sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n"
         "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"sex\"].isna() == False]\n"
         "order = np.random.default_rng(0).permutation(len(df))\n"
         "features = [\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"body_mass_g\"]\n"
         "compare, test = df.iloc[order[:266]], df.iloc[order[266:]]\n"
         "X, y = np.array(compare[features]), np.array(compare[\"sex\"])\n"
         "X_test, y_test = np.array(test[features]), np.array(test[\"sex\"])\n"
         "cv = KFold(5)\n\n")

_LOGISTIC = "cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=cv)"
_KNN = "cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), X, y, cv=cv)"


# --- exercise 1: fold by fold -------------------------------------------------------------------------------------------

def _p(logistic: str = _LOGISTIC, knn: str = _KNN, last: str = "knn - logistic") -> str:
    return _HEAD + f"logistic = {logistic}\nknn = {knn}\n{last}\n"


_P_START = _p("...", "...")
_P = _p()

EXERCISES = {
    "fold-by-fold": CodeBox(
        name="fold-by-fold",
        picture=yours.arrays(),
        start=_P_START,
        solution=_P,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("own-folds", wrong=_p(_LOGISTIC.replace("cv=cv", "cv=5"), _KNN.replace("cv=cv", "cv=5")),
                    en="With `cv=5` scikit-learn chooses its own folds, which keep the share of males the same in every fold: honest too, but other folds than `cv`. Give both `cv=cv`, so that both models are measured on exactly the same penguins.",
                    nl="Met `cv=5` kiest scikit-learn zijn eigen folds, die het aandeel mannetjes in elke fold gelijk houden: ook eerlijk, maar andere folds dan `cv`. Geef ze allebei `cv=cv`, zodat beide modellen op precies dezelfde pinguïns gemeten worden."),
            Mistake("knn-unscaled", wrong=_p(knn="cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=cv)"),
                    en="In your differences, k-NN loses every fold by far. Without standardizing, its distance is almost only body mass. Give it its pipeline: `make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))`.",
                    nl="In je verschillen verliest k-NN elke fold ruim. Zonder standaardiseren is zijn afstand bijna alleen het lichaamsgewicht. Geef hem zijn pipeline: `make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))`."),
            Mistake("reversed", wrong=_p(last="logistic - knn"),
                    en="Your differences are the other way around. The last line is `knn - logistic`, so that a positive number means k-NN did better in that fold.",
                    nl="Je verschillen zijn andersom. De laatste regel is `knn - logistic`, zodat een positief getal betekent dat k-NN het in die fold beter deed."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the 5 fold scores of logistic regression and of k-NN, each in its pipeline, on the folds `cv`.",
                      "nl": "Vervang de twee `...`: de 5 scores van logistische regressie en van k-NN, elk in zijn pipeline, op de folds `cv`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `knn - logistic` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `knn - logistic` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your differences are not those of the course. Do both models get their pipeline with `StandardScaler()` first, and `cv=cv`?",
                      "nl": "Je verschillen zijn niet die van de course. Krijgen beide modellen hun pipeline met eerst `StandardScaler()`, en `cv=cv`?"},
            "correct": {"en": "Per fold: 0.0185, 0.0377, 0.0189, −0.0377 and 0. With about 53 penguins in a fold, 0.0189 is one penguin: k-NN gets one or two more right in three folds, two fewer in one, and the same in one. On average 0.007, less than half a penguin per fold: on these penguins the two are equally good.",
                        "nl": "Per fold: 0,0185, 0,0377, 0,0189, −0,0377 en 0. Met ongeveer 53 pinguïns in een fold is 0,0189 één pinguïn: k-NN heeft er in drie folds één of twee meer goed, in één fold twee minder, en in één evenveel. Gemiddeld 0,007, minder dan een halve pinguïn per fold: op deze pinguïns zijn ze even goed."},
        },
    ),
}


# --- exercise 2: choose k, then test once -------------------------------------------------------------------------------

def _k(score: str = "cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),\n"
                     "                            X, y, cv=cv).mean()",
       pipe: str = "make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best))",
       accuracy: str = "np.mean(pipe.predict(X_test) == y_test)") -> str:
    return (_HEAD + "best = 1\ntop = 0\nfor k in [1, 3, 5, 7, 9, 11, 15, 21, 31]:\n"
            f"    score = {score}\n    if score > top:\n        best = k\n        top = score\n"
            f"pipe = {pipe}\npipe.fit(X, y)\naccuracy = {accuracy}\nbest, accuracy\n")


_K_START = _k("...", "...", "...")
_K = _k()


def _ambiguous(e) -> bool:
    return isinstance(e, ValueError) and "truth value of an array" in str(e)


EXERCISES["choose-k"] = CodeBox(
    name="choose-k",
    start=_K_START,
    solution=_K,
    digits=3,
    first=[_DOTS],
    mistakes=[
        Mistake("chosen-on-test", wrong=_k(score="np.mean(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X, y).predict(X_test) == y_test)"),
                en="You get `{got}`: k was chosen on the test penguins, so the test score is the best of nine tries and too good. Choose k with `cross_val_score` on `X` and `y`, and only use `X_test` once, at the end.",
                nl="Je krijgt `{got}`: k is gekozen op de testpinguïns, dus de testscore is de beste van negen pogingen en te goed. Kies k met `cross_val_score` op `X` en `y`, en gebruik `X_test` maar één keer, aan het eind."),
        Mistake("no-scaler", wrong=_k(score="cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=cv).mean()",
                                      pipe="KNeighborsClassifier(n_neighbors=best)"),
                en="You get `{got}`: without standardizing, the neighbors are chosen almost by body mass alone. Put `StandardScaler()` first in both pipelines.",
                nl="Je krijgt `{got}`: zonder standaardiseren worden de buren bijna alleen op lichaamsgewicht gekozen. Zet `StandardScaler()` eerst in beide pipelines."),
        Mistake("no-mean", error=_ambiguous,
                en="Your code stops with an error: `cross_val_score` gives 5 scores, one per fold, and `if` cannot compare 5 numbers with `top`. Take their mean with `.mean()`.",
                nl="Je code stopt met een fout: `cross_val_score` geeft 5 scores, één per fold, en `if` kan geen 5 getallen met `top` vergelijken. Neem hun gemiddelde met `.mean()`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the three `...`: the cross-validation score of the pipeline with this `k`, the pipeline with the best `k`, and its accuracy on the test penguins.",
                  "nl": "Vervang de drie `...`: de score van cross-validation van de pipeline met deze `k`, de pipeline met de beste `k`, en zijn accuracy op de testpinguïns."},
        "no_value": {"en": "The last line of the box shows nothing. Put `best, accuracy` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `best, accuracy` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "You get `{got}`. Do you score every `k` with `cross_val_score` on `X`, `y` and `cv`, take `.mean()`, and measure the pipeline with `best` once on `X_test`?",
                  "nl": "Je krijgt `{got}`. Geef je elke `k` een score met `cross_val_score` op `X`, `y` en `cv`, neem je `.mean()`, en meet je de pipeline met `best` één keer op `X_test`?"},
        "correct": {"en": "`{got}`: cross-validation chose k = 9, with 0.932, and on the 67 test penguins it gets 0.881, 59 of them right. The test score is lower: 0.932 was the best of nine tries, and some of its lead was luck. 0.881 is the number to report.",
                    "nl": "`{got}`: cross-validation koos k = 9, met 0,932, en op de 67 testpinguïns haalt hij 0,881, 59 goed. De testscore is lager: 0,932 was de beste van negen pogingen, en een deel van die voorsprong was geluk. 0,881 is het getal om te melden."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fold-by-fold", _P_START, "empty", None),
    ("fold-by-fold", _P, "correct", None),
    ("fold-by-fold", _p(_LOGISTIC.replace("cv=cv", "cv=KFold(5)"), _KNN.replace("cv=cv", "cv=KFold(5)")), "correct", None),
    ("fold-by-fold", _HEAD + "lr = make_pipeline(StandardScaler(), LogisticRegression())\nlogistic = cross_val_score(lr, X, y, cv=cv)\nknn = " + _KNN + "\nknn - logistic\n", "correct", None),
    ("fold-by-fold", _p(_LOGISTIC.replace("cv=cv", "cv=5"), _KNN.replace("cv=cv", "cv=5")), "wrong", "own-folds"),
    ("fold-by-fold", _p(knn="cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=cv)"), "wrong", "knn-unscaled"),
    ("fold-by-fold", _p(last="logistic - knn"), "wrong", "reversed"),
    ("fold-by-fold", _p(last="knn.mean() - logistic.mean()"), "wrong", None),
    ("choose-k", _K_START, "empty", None),
    ("choose-k", _K, "correct", None),
    ("choose-k", _k(accuracy="pipe.score(X_test, y_test)"), "correct", None),
    ("choose-k", _k(score="np.mean(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X, y).predict(X_test) == y_test)"), "wrong", "chosen-on-test"),
    ("choose-k", _k(score="cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=cv).mean()", pipe="KNeighborsClassifier(n_neighbors=best)"), "wrong", "no-scaler"),
    ("choose-k", _k(score="cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X, y, cv=cv)"), "error", "no-mean"),
    ("choose-k", _k(pipe="make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))"), "wrong", None),
]
