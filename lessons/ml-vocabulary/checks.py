# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-vocabulary: the features of a table of students who want to know whether they will pass,
and a function that finds the rows a trained model has to predict. And the tables in the picture."""
import csv
import re
from pathlib import Path

from everything_ai.check import CodeBox, FunctionBox, Mistake, same_value

HERE = Path(__file__).parent
HOUSES = "lessons/ml-vocabulary/houses.csv"
STUDENTS = "lessons/ml-vocabulary/students.csv"
WEATHER = "lessons/data-pandas/weather.csv"
STATION = "lessons/data-explore/station.csv"


def picture(labels: list[str]) -> list[tuple]:
    """The houses twice: the feature columns marked, and the label column marked, with ? where it is not known yet."""
    with open(HERE / "houses.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], [[v if v else "?" for v in row] for row in rows[1:]]
    label = header.index("price")
    features = {(i, j) for i in range(len(rows)) for j in range(len(header)) if j != label}
    prices = {(i, label) for i in range(len(rows))}
    return [
        (labels[0], header, rows, features, 'df[["area", "rooms", "garden"]]'),
        (labels[1], header, rows, prices, 'df["price"]'),
    ]


# The lesson is done when these are correct. "spam-label" is the choice question in the lesson.
REQUIRED = ["spam-label", "features", "to-predict"]

_READ = f'df = pd.read_csv("{STUDENTS}")'
_FEATURES_START = "import pandas as pd\n\n" + _READ + "\nfeatures = ...\nfeatures\n"
_FEATURES = "import pandas as pd\n\n" + _READ + '\nfeatures = df[["hours", "sleep"]]\nfeatures'

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _columns(got) -> list[str]:
    """The column names of a table, or the name of one column."""
    kind = type(got).__name__
    return list(got.columns) if kind == "DataFrame" else [got.name] if kind == "Series" else []


def _same_features(a, b) -> bool:
    """The same columns, in any order, with the same values; anything else is compared as usual."""
    if type(a).__name__ == "DataFrame" and type(b).__name__ == "DataFrame":
        return sorted(a.columns) == sorted(b.columns) and same_value(a[sorted(a.columns)], b[sorted(b.columns)])
    return same_value(a, b)


def _shown(value):
    """What a function gave back, in one line: a table by its size and, when it is small, the labels of its rows, in
    words, in the lesson's language."""
    kind = type(value).__name__
    if kind == "DataFrame":
        rows, cols = value.shape
        labels = ", ".join(str(i) for i in value.index) if 0 < rows <= 5 else ""
        which = {"en": f" ({'row' if rows == 1 else 'rows'} {labels})", "nl": f" ({'rij' if rows == 1 else 'rijen'} {labels})"} if labels else {"en": "", "nl": ""}
        return {"en": f"a table of {rows} row{'' if rows == 1 else 's'}{which['en']} and {cols} column{'' if cols == 1 else 's'}",
                "nl": f"een tabel van {rows} {'rij' if rows == 1 else 'rijen'}{which['nl']} en {cols} {'kolom' if cols == 1 else 'kolommen'}"}
    if kind == "Series":
        n = len(value)
        return {"en": f"a column of {n} value{'' if n == 1 else 's'}", "nl": f"een kolom van {n} {'waarde' if n == 1 else 'waarden'}"}
    if type(value).__module__ == "numpy":   # a NumPy number, shown as the plain number it is
        value = value.item()
    return f"`{value!r}`"


_PREDICT_START = "import pandas as pd\n\n\ndef to_predict(path, label):\n    df = pd.read_csv(path)\n    ...\n"
_PREDICT = "import pandas as pd\n\n\ndef to_predict(path, label):\n    df = pd.read_csv(path)\n    return df[df[label].isna()]"


def _predict(new: str) -> str:
    return _PREDICT.replace("    return df[df[label].isna()]", new)


EXERCISES = {
    "features": CodeBox(
        name="features",
        start=_FEATURES_START,
        solution=_FEATURES,
        compare=_same_features,
        keep=["import pandas as pd", _READ],
        first=[_DOTS],
        mistakes=[
            Mistake("brackets", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], tuple),
                    en="For more than one column, the names go in a list: two pairs of square brackets, `df[[\"...\", \"...\"]]`. With one pair, pandas looks for one column with that whole name.",
                    nl="Voor meer dan één kolom staan de namen in een lijst: twee paar vierkante haken, `df[[\"...\", \"...\"]]`. Met één paar zoekt pandas één kolom met die hele naam."),
            Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                    en="The table has no column with that name. Its columns are `hours`, `sleep`, `grade` and `passed`, in small letters, each between quotes.",
                    nl="De tabel heeft geen kolom met die naam. De kolommen zijn `hours`, `sleep`, `grade` en `passed`, in kleine letters, elk tussen aanhalingstekens."),
            Mistake("with-label", matches=lambda got, text: "passed" in _columns(got),
                    en="Your features include `passed`, and that is the label: the answer the model has to predict. A feature is something you know beforehand.",
                    nl="Bij je features zit `passed`, en dat is het label: het antwoord dat het model moet voorspellen. Een feature is iets wat je van tevoren weet."),
            Mistake("with-grade", matches=lambda got, text: "grade" in _columns(got),
                    en="Your features include `grade`, but the grade is only known after the exam, and the model has to predict before it. A feature must be known at the moment you predict.",
                    nl="Bij je features zit `grade`, maar het cijfer is pas na het examen bekend, en het model moet al vóór het examen voorspellen. Een feature moet bekend zijn op het moment dat je voorspelt."),
            Mistake("one-column", matches=lambda got, text: len(_columns(got)) == 1,
                    en="That is one column. Is it the only column that is known before the exam and can help? Take every such column, as a list of names between the square brackets.",
                    nl="Dat is één kolom. Is dat de enige kolom die vóór het examen bekend is en kan helpen? Neem elke zulke kolom, als een lijst namen tussen de vierkante haken."),
            Mistake("names-only", matches=lambda got, text: isinstance(got, list) and all(isinstance(x, str) for x in got),
                    en="That is only a list of names. Put it between the square brackets of `df`, to take those columns from the table.",
                    nl="Dat is alleen een lijst namen. Zet hem tussen de vierkante haken van `df`, om die kolommen uit de tabel te halen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the columns of `df` that are known before the exam: the features.",
                      "nl": "Vervang de `...` door de kolommen van `df` die vóór het examen bekend zijn: de features."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `features` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `features` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be a table with the columns that are known before the exam, for every student. Which columns are those?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet een tabel zijn met, voor elke leerling, de kolommen die vóór het examen bekend zijn. Welke kolommen zijn dat?"},
            "correct": {"en": "The features are `hours` and `sleep`: both known before the exam. `grade` comes only after it, and `passed` is the label.",
                        "nl": "De features zijn `hours` en `sleep`: allebei bekend vóór het examen. `grade` komt pas erna, en `passed` is het label."},
        },
    ),
    "to-predict": FunctionBox(
        name="to-predict",
        function="to_predict",
        start=_PREDICT_START,
        solution=_PREDICT,
        visible=[(HOUSES, "price")],
        # The weather station has missing temperatures but no missing rain: a missing feature does not make a row one to predict.
        hidden=[(STUDENTS, "passed"), (WEATHER, "rain"), (STUDENTS, "grade"), (STATION, "rain")],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("name-in-quotes", error=lambda e: isinstance(e, KeyError) and "label" in str(e),
                    en="`df[\"label\"]` looks for a column that is called `label`. Use the name `label` itself, without quotes: `df[label]`; it holds the name the check gives, such as `\"price\"`.",
                    nl="`df[\"label\"]` zoekt een kolom die `label` heet. Gebruik de naam `label` zelf, zonder aanhalingstekens: `df[label]`; daarin zit de naam die de check geeft, zoals `\"price\"`."),
            Mistake("isna-parentheses", error=lambda e: isinstance(e, TypeError) and "isna() takes" in str(e),
                    en="With `{call}`, Python stops: `.isna` needs its parentheses, `.isna()`, to give the mask.",
                    nl="Met `{call}` stopt Python: `.isna` heeft zijn haakjes nodig, `.isna()`, om het mask te geven."),
            Mistake("fixed-label", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str),
                    en="With `{call}`, your function looks for a column the table does not have: it uses a fixed column name. Use the name `label`, so the function works for every table.",
                    nl="Met `{call}` zoekt je functie een kolom die de tabel niet heeft: hij gebruikt een vaste kolomnaam. Gebruik de naam `label`, dan werkt de functie voor elke tabel."),
            Mistake("any-missing", wrong=_predict("    return df[df.isna().any(axis=1)]"),
                    en="`{call}` gives {got}, and it should give {want}: the rows where any value is missing. Only a missing label makes an example one to predict; a missing feature does not.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de rijen waarin een waarde ontbreekt. Alleen een ontbrekend label maakt een voorbeeld er een om te voorspellen; een ontbrekende feature niet."),
            Mistake("labeled-rows", wrong=_predict("    return df[df[label].notna()]"),
                    en="`{call}` gives {got}: the rows that have a label, the ones a model trains on. The task asks for the other rows: the ones whose label is missing, which a trained model has to predict.",
                    nl="`{call}` geeft {got}: de rijen met een label, waar een model op traint. De opdracht vraagt om de andere rijen: die waarvan het label ontbreekt, en die een getraind model moet voorspellen."),
            Mistake("mask-itself", wrong=_predict("    return df[label].isna()"),
                    en="`{call}` gives the mask itself: `True` or `False` for every row. Put it between the square brackets of `df`, to keep those rows.",
                    nl="`{call}` geeft het mask zelf: `True` of `False` voor elke rij. Zet het tussen de vierkante haken van `df`, om die rijen over te houden."),
            Mistake("label-column", wrong=_predict("    return df[df[label].isna()][label]"),
                    en="`{call}` gives {got}: only the label column of those rows, and it is empty there. Give back the whole rows, with their features: that is what a model predicts from.",
                    nl="`{call}` geeft {got}: alleen de kolom met het label van die rijen, en die is daar leeg. Geef de hele rijen terug, met hun features: daar voorspelt een model uit."),
            Mistake("how-many", wrong=_predict("    return df[label].isna().sum()"),
                    en="`{call}` gives {got}: how many rows lack a label. The task asks for the rows themselves.",
                    nl="`{call}` geeft {got}: hoeveel rijen geen label hebben. De opdracht vraagt om de rijen zelf."),
            Mistake("equals-none", wrong=_predict("    return df[df[label] == None]"),
                    en="`{call}` gives {got}: a missing value is not equal to anything, not even to `None` or another missing value, so `==` finds nothing. Use `.isna()`, as in the lesson *Asking questions of data*.",
                    nl="`{call}` geeft {got}: een ontbrekende waarde is aan niets gelijk, ook niet aan `None` of een andere ontbrekende waarde, dus `==` vindt niets. Gebruik `.isna()`, zoals in de les *Vragen stellen aan data*."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the rows of `df` whose label is missing, given back with `return`.",
                      "nl": "Vervang de `...` door de rijen van `df` waarvan het label ontbreekt, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a path and a column name, and the `def` line of your function has a different number of names between its parentheses. Keep `(path, label)`.",
                          "nl": "De check roept `{call}` aan, met een pad en een kolomnaam, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path, label)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you keep the rows of `df` where the value in the column `label` is missing, with all their columns?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houd je de rijen van `df` over waarin de waarde in de kolom `label` ontbreekt, met al hun kolommen?"},
            "almost": {"en": "Right for the houses. But `{call}` gives {got}, and it should give {want}. Does it work for every table and every label, also when no label is missing?",
                       "nl": "Goed voor de huizen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt hij voor elke tabel en elk label, ook als er geen label ontbreekt?"},
            "correct": {"en": "Your function is right in all {n} cases. The two houses for sale and the three students who still have to take the exam are the examples for inference; the rest is for training.",
                        "nl": "Je functie klopt in alle {n} gevallen. De twee huizen die te koop staan en de drie leerlingen die hun examen nog moeten doen, zijn de voorbeelden voor inference; de rest is voor training."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_S = "import pandas as pd\n\n" + _READ + "\n"
_P = "import pandas as pd\n\n\ndef to_predict(path, label):\n    df = pd.read_csv(path)\n"
CASES = [
    ("features", _FEATURES_START, "empty", None),
    ("features", _FEATURES, "correct", None),
    ("features", _S + 'features = df[["sleep", "hours"]]\nfeatures', "correct", None),
    ("features", _S + 'features = df[["hours", "sleep", "grade"]]\nfeatures', "wrong", "with-grade"),
    ("features", _S + 'features = df[["hours", "sleep", "passed"]]\nfeatures', "wrong", "with-label"),
    ("features", _S + "features = df\nfeatures", "wrong", "with-label"),
    ("features", _S + 'features = df["hours"]\nfeatures', "wrong", "one-column"),
    ("features", _S + 'features = df[["hours"]]\nfeatures', "wrong", "one-column"),
    ("features", _S + 'features = df["passed"]\nfeatures', "wrong", "with-label"),
    ("features", _S + 'features = df["grade"]\nfeatures', "wrong", "with-grade"),
    ("features", _S + 'features = ["hours", "sleep"]\nfeatures', "wrong", "names-only"),
    ("features", _S + 'features = df["hours", "sleep"]\nfeatures', "error", "brackets"),
    ("features", _S + 'features = df[["Hours", "sleep"]]\nfeatures', "error", "no-column"),
    ("features", _S + 'features = df[["hours", "sleep"]]\nprint(features)', "wrong", "print-last"),
    ("to-predict", _PREDICT_START, "empty", None),
    ("to-predict", _PREDICT, "correct", None),
    ("to-predict", _P + "    missing = df[label].isna()\n    return df[missing]", "correct", None),
    ("to-predict", _P + "    return df[df[label].notna()]", "wrong", "labeled-rows"),
    ("to-predict", _P + "    return df[df.isna().any(axis=1)]", "wrong", "any-missing"),
    ("to-predict", _P + "    return df[df.isna().sum(axis=1) > 0]", "wrong", "any-missing"),
    ("to-predict", _P + "    return df[df[label].isna]", "error", "isna-parentheses"),
    ("to-predict", _P + "    return df[~df[label].isna()]", "wrong", "labeled-rows"),
    ("to-predict", _P + "    return df[label].isna()", "wrong", "mask-itself"),
    ("to-predict", _P + "    return df[df[label].isna()][label]", "wrong", "label-column"),
    ("to-predict", _P + "    return df[label].isna().sum()", "wrong", "how-many"),
    ("to-predict", _P + "    return df[df[label] == None]", "wrong", "equals-none"),
    ("to-predict", _P + '    return df[df["label"].isna()]', "error", "name-in-quotes"),
    ("to-predict", _P + '    return df[df["price"].isna()]', "error", "fixed-label"),
]
