# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-pipelines: scikit-learn's pipelines on the Palmer penguins and on the table of noise of
ml-data-leakage. The species of a penguin from its four measurements with 5-nearest neighbors, with and without
standardizing as a step of the pipeline; feature selection as a step, so that cross-validation repeats it in every
fold; and the pipeline fitted on the training penguins and measured once on the test penguins. And the picture: the
score of every fold, and the mean for every k, with and without standardizing."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake
from everything_ai import yours

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
N_TRAIN = 274                # of the 342 penguins with all measurements, shuffled with seed 0: the rest is the test set
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET = "species"
FOLDS = 5
KS = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]   # the stops of the slider: k of k-nearest neighbors
TICKS = [1, 3, 10, 30]


@functools.lru_cache(maxsize=1)
def split():
    """The training and the test penguins as arrays of the four measurements, and their species, as in the worked
    example: seed 0, then 274 and 68 penguins. Read-only, so code that is given them cannot change them."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(0).permutation(len(df))
    train, test = df.iloc[order[:N_TRAIN]], df.iloc[order[N_TRAIN:]]
    out = {"X_train": np.array(train[FOUR]), "y_train": np.array(train[TARGET]),
           "X_test": np.array(test[FOUR]), "y_test": np.array(test[TARGET])}
    for a in out.values():
        a.flags.writeable = False
    return out


@functools.lru_cache(maxsize=2 * len(KS))
def fold_scores(k: int, scaled: bool) -> tuple:
    """The accuracy on each of the 5 folds of the training penguins, with k-nearest neighbors, standardized as the
    first step of a pipeline or not at all."""
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = split()
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)) if scaled else KNeighborsClassifier(n_neighbors=k)
    return tuple(float(v) for v in cross_val_score(model, s["X_train"], s["y_train"], cv=KFold(FOLDS)))


def mean(scores) -> float:
    return sum(scores) / len(scores)


# --- the picture ------------------------------------------------------------------------------------------------------

def picture(k: int, labels: dict[str, str]) -> str:
    """Left: the accuracy on each of the 5 folds with this k, with standardizing in the pipeline (filled dots) and
    without (open dots). Right: the mean over the folds for every k, the same two ways, and this k marked. Drawn as SVG
    text, once for every stop of the slider (L.explore); dragging across the right chart sets k."""
    from everything_ai import chart
    folds = list(range(1, FOLDS + 1))
    p = chart.Panel((0.5, 5.5), (0.6, 1.02), title=labels["title"], xlabel=labels["fold"], ylabel=labels["score"],
                    xticks=folds, yticks=[0.6, 0.7, 0.8, 0.9, 1.0])
    p.points(folds, fold_scores(k, False), "ink-3", 4.0, hollow=True)
    p.points(folds, fold_scores(k, True), "ink", 4.0)
    e = chart.Panel((0.8, 64), (0.6, 1.02), title=labels["every"], xlabel=labels["k"], ylabel=labels["mean"],
                    xticks=TICKS, yticks=[0.6, 0.7, 0.8, 0.9, 1.0], xlog=True)
    e.line([k, k], [0.6, 1.02], "ink-3", 1.2)
    e.line(KS, [mean(fold_scores(j, False)) for j in KS], "ink-3", 1.8)
    e.line(KS, [mean(fold_scores(j, True)) for j in KS], "ink", 2.2)
    e.points([k], [mean(fold_scores(k, False))], "ink-3", 5.0, hollow=True)
    e.points([k], [mean(fold_scores(k, True))], "ink", 5.0)
    e.text(1.0, 0.89, labels["scaled"], "ink", anchor="start")
    e.text(1.0, 0.70, labels["raw"], "ink-3", anchor="start")
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "scaled-first" is the choice question in the lesson.
REQUIRED = ["noise-pipeline", "test-once", "scaled-first"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# --- exercise 1: feature selection as a step ----------------------------------------------------------------------------

_N_HEAD = ("import numpy as np\nfrom sklearn.feature_selection import SelectKBest\n"
           "from sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
           "from sklearn.pipeline import make_pipeline\n\n"
           "rng = np.random.default_rng(6)\nX = rng.random(size=(50, 1000))\ny = rng.permutation([0, 1] * 25)\n\n")


def _n(pipe: str, scores: str = "cross_val_score(pipe, X, y, cv=KFold(5))") -> str:
    return _N_HEAD + f"pipe = {pipe}\nscores = {scores}\nscores.mean()\n"


_N_START = _n("...", "...")
_N = _n("make_pipeline(SelectKBest(k=20), KNeighborsClassifier(n_neighbors=5))")


def _not_a_step(e) -> bool:
    """A model that is not the last step: scikit-learn stops at fit (a step before the last cannot transform), or
    already when cross_val_score looks for the score of the last step."""
    return isinstance(e, TypeError) and ("intermediate steps" in str(e) or "'score' method" in str(e))


EXERCISES = {
    "noise-pipeline": CodeBox(
        name="noise-pipeline",
        picture=yours.steps(),
        start=_N_START,
        solution=_N,
        keep=["rng = np.random.default_rng(6)"],
        # The check also runs the box on another table, so a score that is right by chance is not right there too.
        others=[{"rng": Code("np.random.default_rng(5)")}],
        digits=2,
        first=[_DOTS],
        mistakes=[
            Mistake("selected-first",
                    wrong=_n("make_pipeline(KNeighborsClassifier(n_neighbors=5))",
                             "cross_val_score(pipe, SelectKBest(k=20).fit_transform(X, y), y, cv=KFold(5))"),
                    en="Your score is `{got}`: far better than guessing, on random numbers. `SelectKBest(k=20).fit_transform(X, y)` chooses the columns once, with all 50 people, before the folds: the leak of the previous lesson. Make `SelectKBest(k=20)` the first step of the pipeline, and give `cross_val_score` all of `X`.",
                    nl="Je score is `{got}`: veel beter dan gokken, op willekeurige getallen. `SelectKBest(k=20).fit_transform(X, y)` kiest de kolommen één keer, met alle 50 mensen, vóór de folds: het lek uit de vorige les. Maak `SelectKBest(k=20)` de eerste stap van de pipeline, en geef `cross_val_score` heel `X`."),
            Mistake("no-selection", wrong=_n("make_pipeline(KNeighborsClassifier(n_neighbors=5))"),
                    en="Your score is `{got}`: your pipeline gives all 1000 columns to the neighbors. Put `SelectKBest(k=20)` before the model, as the first step.",
                    nl="Je score is `{got}`: je pipeline geeft alle 1000 kolommen aan de buren. Zet `SelectKBest(k=20)` vóór het model, als eerste stap."),
            Mistake("own-folds", wrong=_n("make_pipeline(SelectKBest(k=20), KNeighborsClassifier(n_neighbors=5))",
                                          "cross_val_score(pipe, X, y)"),
                    en="Your score is `{got}`. Without `cv=KFold(5)`, scikit-learn chooses its own 5 folds, which keep the share of each label the same in every fold. That is honest too, but they are other folds than those of the course. Add `cv=KFold(5)`.",
                    nl="Je score is `{got}`. Zonder `cv=KFold(5)` kiest scikit-learn zijn eigen 5 folds, die het aandeel van elk label in elke fold gelijk houden. Dat is ook eerlijk, maar het zijn andere folds dan die van de course. Voeg `cv=KFold(5)` toe."),
            Mistake("model-first", error=_not_a_step,
                    en="Your code stops with an error: in a pipeline the model comes last, and the steps before it change the data. Put the steps in order: `SelectKBest(k=20)` first, `KNeighborsClassifier(n_neighbors=5)` last.",
                    nl="Je code stopt met een fout: in een pipeline komt het model als laatste, en de stappen ervoor veranderen de data. Zet de stappen op volgorde: `SelectKBest(k=20)` eerst, `KNeighborsClassifier(n_neighbors=5)` als laatste."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: a pipeline of `SelectKBest(k=20)` and `KNeighborsClassifier(n_neighbors=5)`, and its scores with `cross_val_score`.",
                      "nl": "Vervang de twee `...`: een pipeline van `SelectKBest(k=20)` en `KNeighborsClassifier(n_neighbors=5)`, en zijn scores met `cross_val_score`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `scores.mean()` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `scores.mean()` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your score is `{got}`. Is your pipeline `SelectKBest(k=20)` and then `KNeighborsClassifier(n_neighbors=5)`, and do you give `cross_val_score` the pipeline, all of `X` and `y`, and `cv=KFold(5)`?",
                      "nl": "Je score is `{got}`. Is je pipeline `SelectKBest(k=20)` en dan `KNeighborsClassifier(n_neighbors=5)`, en geef je `cross_val_score` de pipeline, heel `X` en `y`, en `cv=KFold(5)`?"},
            "other": {"en": "Right for this table. But on another table, with {where}, your code gives `{got}` and the course's `{want}`. Do you use `X` and `y`, and the steps of the course?",
                      "nl": "Goed voor deze tabel. Maar op een andere tabel, met {where}, geeft je code `{got}` en die van de course `{want}`. Gebruik je `X` en `y`, en de stappen van de course?"},
            "correct": {"en": "An honest score, `{got}`, around guessing: in every fold, `SelectKBest` chose its 20 columns from the 40 training people only. You did not write a loop or a fold: the pipeline did every step in every fold. With the columns chosen first, on all 50 people, the same table gives 0.88; with the way of choosing of the previous lesson it gave 0.96. Both are far too good: `SelectKBest` also weighs how much a column varies within each label, so it keeps somewhat other columns.",
                        "nl": "Een eerlijke score, `{got}`, rond gokken: in elke fold koos `SelectKBest` zijn 20 kolommen alleen uit de 40 trainingsmensen. Je schreef geen loop en geen fold: de pipeline deed elke stap in elke fold. Met de kolommen eerst gekozen, op alle 50 mensen, geeft dezelfde tabel 0,88; met de manier van kiezen uit de vorige les gaf hij 0,96. Allebei veel te goed: `SelectKBest` weegt ook mee hoeveel een kolom binnen elk label varieert, dus houdt hij net andere kolommen."},
        },
    ),
}


# --- exercise 2: fit once, measure once ----------------------------------------------------------------------------------

_T_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.neighbors import KNeighborsClassifier\n"
           "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n"
           "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"body_mass_g\"].isna() == False]\n"
           "order = np.random.default_rng(0).permutation(len(df))\n"
           "features = [\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"body_mass_g\"]\n"
           "train, test = df.iloc[order[:274]], df.iloc[order[274:]]\n"
           "X_train, y_train = np.array(train[features]), np.array(train[\"species\"])\n"
           "X_test, y_test = np.array(test[features]), np.array(test[\"species\"])\n\n")


def _t(pipe: str = "make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))",
       fit: str = "pipe.fit(X_train, y_train)", accuracy: str = "np.mean(pipe.predict(X_test) == y_test)") -> str:
    return _T_HEAD + f"pipe = {pipe}\n{fit}\naccuracy = {accuracy}\naccuracy\n"


_T_START = _t("...", "...", "...")
_T = _t()

EXERCISES["test-once"] = CodeBox(
    name="test-once",
    start=_T_START,
    solution=_T,
    digits=3,
    first=[_DOTS],
    mistakes=[
        Mistake("no-scaler", wrong=_t("make_pipeline(KNeighborsClassifier(n_neighbors=5))"),
                en="Your accuracy is `{got}`: without standardizing, the distance is almost only body mass, in grams. Make `StandardScaler()` the first step of the pipeline.",
                nl="Je accuracy is `{got}`: zonder standaardiseren is de afstand bijna alleen het lichaamsgewicht, in grammen. Maak `StandardScaler()` de eerste stap van de pipeline."),
        Mistake("fit-on-test", wrong=_t(fit="pipe.fit(X_test, y_test)"),
                en="Your accuracy is `{got}`: the pipeline learned from the test penguins themselves, so it has seen the answers. Fit it on `X_train` and `y_train`, and only predict for `X_test`.",
                nl="Je accuracy is `{got}`: de pipeline leerde van de testpinguïns zelf, dus hij heeft de antwoorden gezien. Fit hem op `X_train` en `y_train`, en voorspel alleen voor `X_test`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the three `...`: the pipeline of `StandardScaler()` and `KNeighborsClassifier(n_neighbors=5)`, its fit on the training penguins, and the accuracy on the test penguins.",
                  "nl": "Vervang de drie `...`: de pipeline van `StandardScaler()` en `KNeighborsClassifier(n_neighbors=5)`, de fit op de trainingspinguïns, en de accuracy op de testpinguïns."},
        "no_value": {"en": "The last line of the box shows nothing. Put `accuracy` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `accuracy` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your accuracy is `{got}`. Do you fit the pipeline on `X_train` and `y_train`, and compare `pipe.predict(X_test)` with `y_test`?",
                  "nl": "Je accuracy is `{got}`. Fit je de pipeline op `X_train` en `y_train`, en vergelijk je `pipe.predict(X_test)` met `y_test`?"},
        "correct": {"en": "`{got}`: 67 of the 68 test penguins in the right species, measured once. `pipe.predict` standardized the test penguins with the mean and standard deviation of the 274 training penguins, which `pipe.fit` learned, and then asked their neighbors. This is the score to report.",
                    "nl": "`{got}`: 67 van de 68 testpinguïns in de goede soort, één keer gemeten. `pipe.predict` standaardiseerde de testpinguïns met het gemiddelde en de standaardafwijking van de 274 trainingspinguïns, die `pipe.fit` leerde, en vroeg daarna hun buren. Dit is de score om te melden."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("noise-pipeline", _N_START, "empty", None),
    ("noise-pipeline", _N, "correct", None),
    ("noise-pipeline", _n("make_pipeline(SelectKBest(k=20), KNeighborsClassifier())"), "correct", None),
    ("noise-pipeline", _n("make_pipeline(SelectKBest(k=20), KNeighborsClassifier(n_neighbors=5))", "cross_val_score(pipe, X, y, cv=KFold(n_splits=5))"), "correct", None),
    ("noise-pipeline", _n("make_pipeline(KNeighborsClassifier(n_neighbors=5))", "cross_val_score(pipe, SelectKBest(k=20).fit_transform(X, y), y, cv=KFold(5))"), "wrong", "selected-first"),
    ("noise-pipeline", _n("make_pipeline(KNeighborsClassifier(n_neighbors=5))"), "wrong", "no-selection"),
    ("noise-pipeline", _n("make_pipeline(SelectKBest(k=20), KNeighborsClassifier(n_neighbors=5))", "cross_val_score(pipe, X, y)"), "wrong", "own-folds"),
    ("noise-pipeline", _n("make_pipeline(SelectKBest(k=20), KNeighborsClassifier(n_neighbors=5))", "cross_val_score(pipe, X, y, cv=5)"), "wrong", "own-folds"),
    ("noise-pipeline", _n("make_pipeline(KNeighborsClassifier(n_neighbors=5), SelectKBest(k=20))"), "error", "model-first"),
    ("noise-pipeline", _n("make_pipeline(SelectKBest(k=50), KNeighborsClassifier(n_neighbors=5))"), "wrong", None),
    ("test-once", _T_START, "empty", None),
    ("test-once", _T, "correct", None),
    ("test-once", _t(accuracy="pipe.score(X_test, y_test)"), "correct", None),
    ("test-once", _t(fit="pipe = pipe.fit(X_train, y_train)"), "correct", None),
    ("test-once", _t("make_pipeline(KNeighborsClassifier(n_neighbors=5))"), "wrong", "no-scaler"),
    ("test-once", _t(fit="pipe.fit(X_test, y_test)"), "wrong", "fit-on-test"),
    ("test-once", _t(accuracy="np.mean(pipe.predict(X_train) == y_train)"), "wrong", None),
]
