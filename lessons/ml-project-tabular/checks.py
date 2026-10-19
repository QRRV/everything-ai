# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the review lesson ml-project-tabular: the workflow on the Palmer penguins. Logistic regression on the
four measurements, standardized with the training set, measured on the validation set; and the penguins it gets
wrong. And the picture of the training penguins by bill length and depth."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake, run_code, same_value

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
N_TRAIN, N_VAL = 206, 68     # of the 342 penguins with all four measurements; the other 68 are the test set
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


@functools.lru_cache(maxsize=1)
def split():
    """The training, validation and test set, as in the worked example: seed 0, then 206, 68 and 68 penguins."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(0).permutation(len(df))
    return df.iloc[order[:N_TRAIN]], df.iloc[order[N_TRAIN:N_TRAIN + N_VAL]], df.iloc[order[N_TRAIN + N_VAL:]]


def picture(labels: dict[str, str]) -> str:
    """The training penguins by bill length and bill depth, one colour and shape per species, with a key."""
    from everything_ai import chart
    train, _, _ = split()
    p = chart.Panel((30, 60), (12, 22), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for k, s in enumerate(SPECIES):
        rows = train[train["species"] == s]
        p.group(rows["bill_length_mm"], rows["bill_depth_mm"], k)
    for k, s in enumerate(SPECIES):   # the key, bottom left, where there are no penguins
        p.group([32], [14.1 - 0.8 * k], k)
        p.text(33, 14.1 - 0.8 * k - 0.12, s, "ink-2", anchor="start")
    return chart.figure([p], alt=labels["alt"], width=520, height=380)


# The lesson is done when these are correct. "proxy" is the choice question in the lesson.
REQUIRED = ["model", "mistakes", "proxy"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_FEATURES = 'features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]'
_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n\n"
         'df = pd.read_csv("data/penguins.csv")\ndf = df[df["body_mass_g"].isna() == False]\n'
         "order = np.random.default_rng(0).permutation(len(df))\ntrain = df.iloc[order[:206]]\nval = df.iloc[order[206:274]]\n"
         + _FEATURES + "\n\n")
_MODEL = ("mean = train[features].mean()\nstd = train[features].std()\nmodel = LogisticRegression()\n"
          'model.fit((train[features] - mean) / std, train["species"])\npred = model.predict((val[features] - mean) / std)\n')
_ACC = 'accuracy = np.mean(pred == val["species"])\n'
_M_START = _HEAD + "...\naccuracy\n"
_M = _HEAD + _MODEL + _ACC + "accuracy"
_W_START = _HEAD + _MODEL + "\n...\nwrong\n"
_W = _HEAD + _MODEL + '\nwrong = val[pred != val["species"]]\nwrong'


def _m(model: str, acc: str = _ACC) -> str:
    return _HEAD + model + acc + "accuracy"


def _w(new: str) -> str:
    return _HEAD + _MODEL + "\n" + new + "\nwrong"


# The mean taken off, but not divided by the standard deviation. Its accuracy, 67 of 68, is also what the test set, a
# training on `val` or standardizing `val` with its own numbers give; so the check only says "mean only" when no
# line divides by a standard deviation. scikit-learn warns that this version does not converge: the learner did not
# write it, so the check works it out once with that warning off, instead of running it after every wrong answer.
_MEAN_ONLY = _m(_MODEL.replace("(train[features] - mean) / std", "train[features] - mean").replace("(val[features] - mean) / std", "val[features] - mean"))


@functools.lru_cache(maxsize=1)
def _mean_only_value():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_code(_MEAN_ONLY).value


def _mean_only(got, text: str) -> bool:
    return re.search(r"^[^#\n]*/[^\n]*std", text, re.MULTILINE) is None and same_value(got, _mean_only_value())


_KEEP = ['df = df[df["body_mass_g"].isna() == False]', "order = np.random.default_rng(0).permutation(len(df))",
         "train = df.iloc[order[:206]]", "val = df.iloc[order[206:274]]", _FEATURES]
_OTHER_FEATURES = Code('["bill_length_mm", "bill_depth_mm"]')
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}
_KEPT = {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other features by itself.",
         "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere features."}

EXERCISES = {
    "model": CodeBox(
        name="model",
        start=_M_START,
        solution=_M,
        keep=_KEEP,
        # The check also runs the box with other features: a number typed in by hand does not change with them.
        others=[{"features": _OTHER_FEATURES}],
        first=[
            _DOTS,
            Mistake("val-stats", matches=lambda got, text: re.search(r"^[^#\n]*\b(val|df)\[\s*features\s*\]\s*\.\s*(mean|std)\s*\(", text, re.MULTILINE) is not None,
                    en="You take a mean or standard deviation of the validation set, or of the whole table. Standardize every set with the mean and standard deviation of the training set only: the validation set stands for new penguins, which you do not have yet when you train.",
                    nl="Je neemt een gemiddelde of standaardafwijking van de validation set, of van de hele tabel. Standaardiseer elke set alleen met het gemiddelde en de standaardafwijking van de training set: de validation set staat voor nieuwe pinguïns, die je nog niet hebt als je traint."),
            Mistake("test-set", matches=lambda got, text: re.search(r"^[^#\n]*(\btest\b|order\[\s*274\s*:)", text, re.MULTILINE) is not None,
                    en="You use the test set. It comes in once, at the very end, when everything is chosen; until then you measure and choose on the validation set, `val`.",
                    nl="Je gebruikt de test set. Die komt pas aan het eind, één keer, als alles gekozen is; tot dan meet en kies je op de validation set, `val`."),
        ],
        form=[Mistake("not-scaled", state="almost", matches=lambda got, text: re.search(r"^[^#\n]*std\s*\(", text, re.MULTILINE) is None,
                      en="The right accuracy, but without standardizing. scikit-learn then warns that it did not converge: body mass, around 4000 grams, is far bigger than the other features. Standardize with the mean and standard deviation of the training set, as in the worked example.",
                      nl="De goede accuracy, maar zonder te standaardiseren. scikit-learn waarschuwt dan dat de training niet convergeerde: het lichaamsgewicht, rond de 4000 gram, is veel groter dan de andere features. Standaardiseer met het gemiddelde en de standaardafwijking van de training set, zoals in het uitgewerkte voorbeeld.")],
        mistakes=[
            Mistake("rounded", state="almost", matches=lambda got, text: isinstance(got, float) and abs(got - 66 / 68) < 0.006,
                    en="Your last line gives `{got}`: close, but rounded or typed in. Let Python work out the accuracy from `pred` and `val[\"species\"]`.",
                    nl="Je laatste regel geeft `{got}`: dichtbij, maar afgerond of ingetypt. Laat Python de accuracy uitrekenen met `pred` en `val[\"species\"]`."),
            Mistake("val-raw", wrong=_m(_MODEL.replace("model.predict((val[features] - mean) / std)", "model.predict(val[features])")),
                    en="Your last line gives about `0.37`: the model learned on standardized features, and predicts on raw ones. Standardize the validation set too, with the same `mean` and `std`.",
                    nl="Je laatste regel geeft ongeveer `0.37`: het model leerde op gestandaardiseerde features, en voorspelt op ruwe. Standaardiseer de validation set ook, met dezelfde `mean` en `std`."),
            Mistake("mean-only", matches=_mean_only,
                    en="Your last line gives about `0.99`: you take the mean off, but do not divide by the standard deviation. Standardizing is both: `(x - mean) / std`.",
                    nl="Je laatste regel geeft ongeveer `0.99`: je haalt het gemiddelde eraf, maar deelt niet door de standaardafwijking. Standaardiseren is allebei: `(x - mean) / std`."),
            Mistake("on-train", wrong=_HEAD + _MODEL + 'accuracy = np.mean(model.predict((train[features] - mean) / std) == train["species"])\naccuracy',
                    en="Your last line gives about `0.99`: the accuracy on the training set. The model learned from those penguins; measure it on the validation set, `val`.",
                    nl="Je laatste regel geeft ongeveer `0.99`: de accuracy op de training set. Het model leerde van die pinguïns; meet het op de validation set, `val`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with step 5: standardize with the training set, let `LogisticRegression` learn, predict the validation set, and put the accuracy in `accuracy`.",
                      "nl": "Vervang de `...` door stap 5: standaardiseer met de training set, laat `LogisticRegression` leren, voorspel de validation set, en zet de accuracy in `accuracy`."},
            "cleared": _CLEARED,
            "keep": _KEPT,
            "no_value": {"en": "The last line of the box shows nothing. Put `accuracy` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `accuracy` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you standardize both sets with the mean and standard deviation of the training set, train on `train`, and compare the predictions for `val` with `val[\"species\"]`?",
                      "nl": "Je laatste regel geeft `{got}`. Standaardiseer je beide sets met het gemiddelde en de standaardafwijking van de training set, train je op `train`, en vergelijk je de voorspellingen voor `val` met `val[\"species\"]`?"},
            "other": {"en": "Right for these features. But with {where}, your last line gives `{got}`, and it should give about `0.94`. Do you work it out with `features` itself?",
                      "nl": "Goed voor deze features. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `0.94` zijn. Reken je het uit met `features` zelf?"},
            "other_error": {"en": "Right for these features. But with {where}, your code stops with an error. Do you work it out with `features` itself?",
                            "nl": "Goed voor deze features. Maar met {where} stopt je code met een fout. Reken je het uit met `features` zelf?"},
            "correct": {"en": "About 0.97: 66 of the 68 penguins of the validation set, against 0.44 for the baseline, and above the goal of 0.95. With two features it was 0.84. The check also tried other features.",
                        "nl": "Ongeveer 0,97: 66 van de 68 pinguïns van de validation set, tegen 0,44 voor de baseline, en boven het doel van 0,95. Met twee features was het 0,84. De check probeerde ook andere features."},
        },
    ),
    "mistakes": CodeBox(
        name="mistakes",
        start=_W_START,
        solution=_W,
        keep=_KEEP,
        others=[{"features": _OTHER_FEATURES}],
        first=[_DOTS],
        mistakes=[
            Mistake("right-ones", wrong=_w('wrong = val[pred == val["species"]]'),
                    en="Your last line gives the penguins the model gets right: `==` keeps the rows where prediction and species are the same. You want the others: `!=`.",
                    nl="Je laatste regel geeft de pinguïns die het model goed heeft: `==` houdt de rijen waar voorspelling en soort gelijk zijn. Je wilt de andere: `!=`."),
            Mistake("mask", wrong=_w('wrong = pred != val["species"]'),
                    en="Your last line gives `True` or `False` for each penguin: the mask itself. Put it between the square brackets of `val`, so that you keep the rows where it is `True`.",
                    nl="Je laatste regel geeft `True` of `False` voor elke pinguïn: het masker zelf. Zet het tussen de vierkante haken van `val`, zodat je de rijen houdt waar het `True` is."),
            Mistake("predictions", matches=lambda got, text: type(got).__name__ == "ndarray" and got.dtype.kind in "OUS",
                    en="Your last line gives predictions, not penguins. Use the comparison as a mask on `val`, so that you keep whole rows: `val[...]`.",
                    nl="Je laatste regel geeft voorspellingen, geen pinguïns. Gebruik de vergelijking als masker op `val`, zodat je hele rijen houdt: `val[...]`."),
            Mistake("all", wrong=_w("wrong = val"),
                    en="Your last line gives all 68 penguins of the validation set. Keep only the rows where `pred` differs from `val[\"species\"]`.",
                    nl="Je laatste regel geeft alle 68 pinguïns van de validation set. Houd alleen de rijen waar `pred` verschilt van `val[\"species\"]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: put in `wrong` the rows of `val` where `pred` differs from the species.",
                      "nl": "Vervang de `...` door één regel: zet in `wrong` de rijen van `val` waar `pred` verschilt van de soort."},
            "cleared": _CLEARED,
            "keep": _KEPT,
            "no_value": {"en": "The last line of the box shows nothing. Put `wrong` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `wrong` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line does not give the penguins the model gets wrong. Is `wrong` the rows of `val` where `pred` differs from `val[\"species\"]`?",
                      "nl": "Je laatste regel geeft niet de pinguïns die het model fout heeft. Is `wrong` de rijen van `val` waar `pred` verschilt van `val[\"species\"]`?"},
            "other": {"en": "Right for these features. But with {where}, your last line gives other penguins than it should. Do you work it out from `pred` and `val[\"species\"]`, instead of typing the rows in?",
                      "nl": "Goed voor deze features. Maar met {where} geeft je laatste regel andere pinguïns dan zou moeten. Reken je het uit met `pred` en `val[\"species\"]`, in plaats van de rijen in te typen?"},
            "other_error": {"en": "Right for these features. But with {where}, your code stops with an error. Do you work it out from `pred` and `val[\"species\"]`, instead of typing the rows in?",
                            "nl": "Goed voor deze features. Maar met {where} stopt je code met een fout. Reken je het uit met `pred` en `val[\"species\"]`, in plaats van de rijen in te typen?"},
            "correct": {"en": "Two penguins: both Chinstraps from Dream, with bills of 42.4 and 43.5 mm, short for a Chinstrap. The model took both for an Adelie. The check also tried other features.",
                        "nl": "Twee pinguïns: allebei Chinstraps van Dream, met snavels van 42,4 en 43,5 mm, kort voor een Chinstrap. Het model hield ze allebei voor een Adelie. De check probeerde ook andere features."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("model", _M_START, "empty", None),
    ("model", _M, "correct", None),
    ("model", _m(_MODEL, 'accuracy = (pred == val["species"]).mean()\n'), "correct", None),
    ("model", _m(_MODEL.replace("model = LogisticRegression()\nmodel.fit(", "model = LogisticRegression().fit(")), "correct", None),
    ("model", _m("Z_train = (train[features] - train[features].mean()) / train[features].std()\n"
                 "Z_val = (val[features] - train[features].mean()) / train[features].std()\n"
                 'model = LogisticRegression()\nmodel.fit(Z_train, train["species"])\npred = model.predict(Z_val)\n'), "correct", None),
    ("model", _m(_MODEL.replace("(val[features] - mean) / std", "(val[features] - val[features].mean()) / val[features].std()")), "wrong", "val-stats"),
    ("model", _m(_MODEL.replace("model.predict((val[features] - mean) / std)", "model.predict(val[features])")), "wrong", "val-raw"),
    ("model", _m(_MODEL.replace("(train[features] - mean) / std", "train[features] - mean").replace("(val[features] - mean) / std", "val[features] - mean")), "wrong", "mean-only"),
    ("model", _HEAD + _MODEL + 'accuracy = np.mean(model.predict((train[features] - mean) / std) == train["species"])\naccuracy', "wrong", "on-train"),
    ("model", _m('model = LogisticRegression()\nmodel.fit(train[features], train["species"])\npred = model.predict(val[features])\n'), "almost", "not-scaled"),
    ("model", _m(_MODEL.replace("mean = train[features].mean()\nstd = train[features].std()", "mean = df[features].mean()\nstd = df[features].std()")), "wrong", "val-stats"),
    ("model", _HEAD + "test = df.iloc[order[274:]]\n" + _MODEL.replace("val[features]", "test[features]") + 'accuracy = np.mean(pred == test["species"])\naccuracy', "wrong", "test-set"),
    ("model", _m(_MODEL.replace("model.predict((val[features] - mean) / std)", "model.predict((X_val - X_val.mean()) / X_val.std())").replace("model = LogisticRegression()", "X_val = val[features]\nmodel = LogisticRegression()")), "wrong", None),
    ("model", _m(_MODEL.replace("(train[features] - mean) / std", "train[features] / std").replace("(val[features] - mean) / std", "val[features] / std")), "wrong", None),
    ("model", _m("Z_train = (train[features] - train[features].mean()) / train[features].std(ddof=0)\n"
                 "Z_val = (val[features] - train[features].mean()) / train[features].std(ddof=0)\n"
                 'model = LogisticRegression()\nmodel.fit(Z_train, train["species"])\npred = model.predict(Z_val)\n'), "correct", None),
    ("model", _m(_MODEL, "accuracy = 0.97\n"), "almost", "rounded"),
    ("model", _m(_MODEL, "accuracy = 66 / 68\n"), "almost", "other-value"),
    ("model", _M.replace(_FEATURES, 'features = ["bill_length_mm", "bill_depth_mm"]'), "wrong", "changed-line"),
    ("mistakes", _W_START, "empty", None),
    ("mistakes", _W, "correct", None),
    ("mistakes", _w('wrong = val[val["species"] != pred]'), "correct", None),
    ("mistakes", _w('wrong = val[pred == val["species"]]'), "wrong", "right-ones"),
    ("mistakes", _w('wrong = pred[pred != val["species"]]'), "wrong", "predictions"),
    ("mistakes", _w("wrong = val"), "wrong", "all"),
    ("mistakes", _w('wrong = pred != val["species"]'), "wrong", "mask"),
    ("mistakes", _w('wrong = (pred != val["species"]).to_numpy()'), "wrong", "mask"),
    ("mistakes", _w('wrong = np.where(pred != val["species"])[0]'), "wrong", None),
    ("mistakes", _w("wrong = val.loc[[296, 340]]"), "almost", "other-value"),
]
