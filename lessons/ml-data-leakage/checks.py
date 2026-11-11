# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-data-leakage: feature selection before cross-validation on a table of pure noise, after
Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, section 7.10.2. 50 people, columns of random
numbers between 0 and 1, and a label of 0 or 1 drawn at random, half of each; the 20 columns whose mean differs most
between the two labels; 5-fold cross-validation with 5-nearest neighbors. Choosing the columns inside each fold, which
columns leak, and why a group stays on one side of a split. And the picture: for every number of columns, the
cross-validation score of ten such tables with the columns chosen on all 50 people and inside each fold, and the gaps
of the chosen columns among the 50 and among new people."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

PEOPLE = 50                 # people in a table, half with label 1
NEW = 1000                  # new people, to see what a model really gets right
KEEP = 20                   # the columns that are kept
NEIGHBORS = 5
FOLDS = 5
TABLES = 10                 # tables with their own random numbers, seeds 0 to 9, for the spread in the picture
EXAMPLE = 6                 # the table of the worked example: seed 6
COLUMNS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]   # the stops of the slider
TICKS = [10, 100, 1000, 10000]
SIDE = 1.15                 # in the picture, the two kinds of score of one number of columns stand this factor apart


def table(seed: int, columns: int):
    """A table of noise, as the worked example makes it: a generator with this seed draws the numbers, then the labels."""
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.random(size=(PEOPLE, columns))
    y = rng.permutation([0, 1] * (PEOPLE // 2))
    return X, y


def select(X, y, k: int = KEEP):
    """The lesson's feature selection: the k columns whose mean differs most between label 1 and label 0."""
    import numpy as np
    gap = abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
    return np.argsort(gap)[-k:]


def gaps(X, y, cols):
    """How much the mean of each of these columns differs between label 1 and label 0."""
    return abs(X[y == 1][:, cols].mean(axis=0) - X[y == 0][:, cols].mean(axis=0))


def score(X, y, leaky: bool) -> float:
    """The share of the people that 5-fold cross-validation with 5-NN gets right, with the columns chosen on all the
    people (leaky) or again in every fold, from its training people only."""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    rows = np.arange(PEOPLE)
    size = PEOPLE // FOLDS
    right = 0
    cols = select(X, y)
    for f in range(FOLDS):
        val = rows[size * f:size * (f + 1)]
        train = np.concatenate([rows[:size * f], rows[size * (f + 1):]])
        if not leaky:
            cols = select(X[train], y[train])
        model = KNeighborsClassifier(n_neighbors=NEIGHBORS).fit(X[train][:, cols], y[train])
        right = right + int(np.sum(model.predict(X[val][:, cols]) == y[val]))
    return right / PEOPLE


def spread(columns: int, leaky: bool) -> list:
    """The scores of the ten tables with this many columns. All of them take two seconds, so the picture uses MEASURED,
    which tests/test_ml_data_leakage.py compares with this."""
    return [score(*table(seed, columns), leaky) for seed in range(TABLES)]


# spread(columns, leaky) for every number of columns of the slider, tables 0 to 9 in that order.
MEASURED = {
    True: {10: [0.48, 0.5, 0.56, 0.5, 0.42, 0.48, 0.54, 0.54, 0.54, 0.46], 20: [0.44, 0.34, 0.3, 0.48, 0.62, 0.38, 0.48, 0.34, 0.42, 0.44],
           50: [0.76, 0.6, 0.66, 0.6, 0.68, 0.52, 0.48, 0.5, 0.72, 0.5], 100: [0.58, 0.94, 0.66, 0.7, 0.68, 0.6, 0.78, 0.68, 0.82, 0.82],
           200: [0.78, 0.78, 0.8, 0.86, 0.78, 0.76, 0.72, 0.82, 0.7, 0.7], 500: [0.9, 0.86, 0.86, 0.86, 0.8, 0.88, 0.9, 0.86, 0.96, 0.82],
           1000: [0.96, 0.9, 0.88, 0.9, 0.88, 0.9, 0.96, 0.92, 0.92, 0.82], 2000: [0.94, 0.94, 1.0, 0.9, 0.92, 0.94, 0.94, 0.9, 0.98, 0.92],
           5000: [0.9, 0.98, 0.98, 0.96, 0.94, 0.9, 0.96, 0.92, 0.94, 0.9], 10000: [0.92, 0.96, 0.96, 0.98, 0.94, 0.92, 0.98, 0.96, 0.92, 0.96]},
    False: {10: [0.48, 0.5, 0.56, 0.5, 0.42, 0.48, 0.54, 0.54, 0.54, 0.46], 20: [0.44, 0.34, 0.3, 0.48, 0.62, 0.38, 0.48, 0.34, 0.42, 0.44],
            50: [0.66, 0.46, 0.58, 0.46, 0.54, 0.34, 0.4, 0.36, 0.56, 0.52], 100: [0.42, 0.52, 0.56, 0.5, 0.5, 0.42, 0.58, 0.58, 0.56, 0.62],
            200: [0.6, 0.64, 0.6, 0.48, 0.44, 0.56, 0.46, 0.38, 0.38, 0.46], 500: [0.54, 0.44, 0.58, 0.4, 0.48, 0.6, 0.6, 0.56, 0.64, 0.4],
            1000: [0.44, 0.38, 0.5, 0.54, 0.46, 0.62, 0.52, 0.42, 0.38, 0.44], 2000: [0.56, 0.4, 0.52, 0.62, 0.48, 0.38, 0.52, 0.34, 0.54, 0.44],
            5000: [0.52, 0.4, 0.42, 0.38, 0.42, 0.56, 0.44, 0.4, 0.44, 0.54], 10000: [0.48, 0.48, 0.32, 0.58, 0.6, 0.48, 0.48, 0.42, 0.3, 0.32]},
}


@functools.lru_cache(maxsize=len(COLUMNS))
def chosen_gaps(columns: int):
    """For the table of the worked example with this many columns: the gaps of the kept columns among its 50 people,
    largest first, and the gaps of the same columns among 1000 new people. A new person's numbers in a column are fresh
    random numbers whatever the column, so the new people get only those columns, from a generator of their own."""
    import numpy as np
    X, y = table(EXAMPLE, columns)
    cols = select(X, y)[::-1]
    rng = np.random.default_rng(1000 + columns)
    new_X = rng.random(size=(NEW, len(cols)))
    new_y = rng.permutation([0, 1] * (NEW // 2))
    return gaps(X, y, cols), gaps(new_X, new_y, np.arange(len(cols)))


# --- the picture ------------------------------------------------------------------------------------------------------

def picture(columns: int, labels: dict[str, str]) -> str:
    """Left: the kept columns of the worked example's table, largest gap first: a filled dot for the gap among its 50
    people, what the selection saw, and an open dot for the gap among 1000 new people. Right: for every number of
    columns, the cross-validation scores of the ten tables, with the columns chosen on all 50 people (filled dots, ink)
    and inside each fold (open dots, grey), a line from the lowest to the highest, guessing at 0.5, and this number
    marked. Drawn as SVG text, once for every stop of the slider (L.explore); dragging across the right chart sets it."""
    from everything_ai import chart
    seen, new = chosen_gaps(columns)
    ranks = list(range(1, len(seen) + 1))
    p = chart.Panel((0, 21), (0, 0.36), title=labels["title"], xlabel=labels["rank"], ylabel=labels["gap"],
                    xticks=[1, 5, 10, 15, 20], yticks=[0, 0.1, 0.2, 0.3])
    p.points(ranks, new, "ink-3", 3.4, hollow=True)
    p.points(ranks, seen, "ink", 3.6)
    e = chart.Panel((7, 14000), (0, 1.04), title=labels["every"], xlabel=labels["columns"], ylabel=labels["score"],
                    xticks=TICKS, yticks=[0, 0.25, 0.5, 0.75, 1], xlog=True)
    e.line([7, 14000], [0.5, 0.5], "ink-3", 1.2, dashed=True)
    e.line([columns, columns], [0.2, 1.04], "ink-3", 1.2)
    for n in COLUMNS:
        for leaky, role, hollow, x in ((False, "ink-3", True, n / SIDE), (True, "ink", False, n * SIDE)):
            m = MEASURED[leaky][n]
            e.line([x, x], [min(m), max(m)], role, 1.4)
            e.points([x] * len(m), m, role, 2.8, hollow=hollow)
    e.points([columns * SIDE], [MEASURED[True][columns][EXAMPLE]], "ink", 5.5)
    e.points([columns / SIDE], [MEASURED[False][columns][EXAMPLE]], "ink-3", 5.5, hollow=True)
    e.text(9, 0.15, labels["leaky"], "ink", anchor="start")
    e.text(9, 0.06, labels["honest"], "ink-3", anchor="start")
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "same-patient" is the choice question in the lesson.
REQUIRED = ["inside-folds", "known-then", "same-patient"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


# --- exercise 1: inside the folds -------------------------------------------------------------------------------------

_F_HEAD = ("import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\n\n"
           "rng = np.random.default_rng(2)\nX = rng.random(size=(50, 1000))\ny = rng.permutation([0, 1] * 25)\n\n\n"
           "def select(X, y, k):\n    gap = abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))\n    return np.argsort(gap)[-k:]\n\n\n"
           "rows = np.arange(50)\nright = 0\nfor f in range(5):\n    val = rows[10 * f:10 * (f + 1)]\n"
           "    train = np.concatenate([rows[:10 * f], rows[10 * (f + 1):]])\n")
_F_TAIL = ("    model = KNeighborsClassifier(n_neighbors=5).fit(X[train][:, cols], y[train])\n"
           "    right = right + np.sum(model.predict(X[val][:, cols]) == y[val])\nright / 50\n")


def _f(cols: str) -> str:
    return _F_HEAD + f"    cols = {cols}\n" + _F_TAIL


_F_START = _f("...")
_F = _f("select(X[train], y[train], 20)")


def _index_error(e) -> bool:
    return isinstance(e, IndexError) and "boolean" in str(e)


EXERCISES = {
    "inside-folds": CodeBox(
        name="inside-folds",
        picture=yours.steps(),
        start=_F_START,
        solution=_F,
        keep=["rng = np.random.default_rng(2)"],
        # The check also runs the box on another table: a score that is right by chance, such as with 19 or 21 columns,
        # is not right there too.
        others=[{"rng": Code("np.random.default_rng(5)")}],
        first=[_DOTS],
        mistakes=[
            Mistake("all-people", wrong=_f("select(X, y, 20)"),
                    en="Your score is `{got}`: far better than guessing, on random numbers. `select(X, y, 20)` chooses the columns with the labels of all 50 people, so the validation fold helped choose them. Choose them from the training people of the fold: `X[train]` and `y[train]`.",
                    nl="Je score is `{got}`: veel beter dan gokken, op willekeurige getallen. `select(X, y, 20)` kiest de kolommen met de labels van alle 50 mensen, dus de validation-fold hielp ze kiezen. Kies ze uit de trainingsmensen van de fold: `X[train]` en `y[train]`."),
            Mistake("validation-people", wrong=_f("select(X[val], y[val], 20)"),
                    en="Your score is `{got}`. `X[val]` and `y[val]` are the people the fold checks on: now they alone choose the columns. Choose them from the training people of the fold: `X[train]` and `y[train]`.",
                    nl="Je score is `{got}`. `X[val]` en `y[val]` zijn de mensen waarop de fold controleert: nu kiezen zij alleen de kolommen. Kies ze uit de trainingsmensen van de fold: `X[train]` en `y[train]`."),
            Mistake("other-people", error=_index_error,
                    en="Your code stops with an error: `select` gets the rows of `X` and the labels in `y` of different people. Give it both of the training people of the fold: `X[train]` and `y[train]`.",
                    nl="Je code stopt met een fout: `select` krijgt de rijen van `X` en de labels in `y` van verschillende mensen. Geef hem beide van de trainingsmensen van de fold: `X[train]` en `y[train]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...`: the 20 columns that `select` chooses, from the training people of this fold only.",
                      "nl": "Vervang de `...`: de 20 kolommen die `select` kiest, alleen uit de trainingsmensen van deze fold."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `right / 50` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `right / 50` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your score is `{got}`. Do you let `select` choose 20 columns, with `X[train]` and `y[train]`?",
                      "nl": "Je score is `{got}`. Laat je `select` 20 kolommen kiezen, met `X[train]` en `y[train]`?"},
            "other": {"en": "Right for this table. But on another table, with {where}, your code gives `{got}` and the course's `{want}`. Does `select` keep 20 columns, of the training people?",
                      "nl": "Goed voor deze tabel. Maar op een andere tabel, met {where}, geeft je code `{got}` en die van de course `{want}`. Houdt `select` 20 kolommen, van de trainingsmensen?"},
            "correct": {"en": "An honest score: `{got}`, about what guessing gives, as it should on random numbers. With 50 people, one person more or less right is 0.02. With the columns chosen on all 50 people, the same code gives 0.88.",
                        "nl": "Een eerlijke score: `{got}`, ongeveer wat gokken geeft, zoals het hoort op willekeurige getallen. Met 50 mensen is één persoon meer of minder goed 0,02. Met de kolommen gekozen op alle 50 mensen geeft dezelfde code 0,88."},
        },
    ),
}


# --- exercise 2: known on 1 September, or leaking? ---------------------------------------------------------------------

EXERCISES["known-then"] = Sort(
    name="known-then",
    groups={
        "known": {"en": "Known on 1 September: may be a feature", "nl": "Bekend op 1 september: mag een feature zijn"},
        "leaks": {"en": "Only known later, or made from the answer: leaks", "nl": "Pas later bekend, of gemaakt uit het antwoord: lekt"},
    },
    items={
        "grades": {"en": "**Last year's grades.** The mean of the student's grades in the previous school year.",
                   "nl": "**Cijfers van vorig jaar.** Het gemiddelde van de cijfers van de leerling in het vorige schooljaar."},
        "distance": {"en": "**Distance to school.** How far from school the student lives, in kilometers.",
                     "nl": "**Afstand tot school.** Hoe ver van school de leerling woont, in kilometers."},
        "subjects": {"en": "**Subjects.** How many subjects the student takes this year, chosen last spring, before the summer holidays.",
                     "nl": "**Vakken.** Hoeveel vakken de leerling dit jaar volgt, vorig voorjaar gekozen, voor de zomervakantie."},
        "absences": {"en": "**Days absent.** How many days the student missed this school year, counted at the end of June.",
                     "nl": "**Dagen afwezig.** Hoeveel dagen de leerling dit schooljaar miste, eind juni geteld."},
        "summer": {"en": "**Summer course.** Whether the student took the school's summer course for students who failed.",
                   "nl": "**Zomercursus.** Of de leerling de zomercursus van school volgde voor leerlingen die zakten."},
        "diploma": {"en": "**Diploma date.** The date on which the student got the diploma; empty if they did not.",
                    "nl": "**Datum diploma.** De datum waarop de leerling het diploma kreeg; leeg als dat niet gebeurde."},
    },
    solution={"grades": "known", "distance": "known", "subjects": "known", "absences": "leaks", "summer": "leaks", "diploma": "leaks"},
    nudge={
        "grades": {"en": "When were those grades given: before or after 1 September?",
                   "nl": "Wanneer zijn die cijfers gegeven: voor of na 1 september?"},
        "distance": {"en": "Does the school know where a student lives at the start of the year?",
                     "nl": "Weet de school aan het begin van het jaar waar een leerling woont?"},
        "subjects": {"en": "When were the subjects chosen?",
                     "nl": "Wanneer zijn de vakken gekozen?"},
        "absences": {"en": "On 1 September, how many days of this school year can you have counted?",
                     "nl": "Hoeveel dagen van dit schooljaar kun je op 1 september geteld hebben?"},
        "summer": {"en": "Who takes that course, and when do you know who failed?",
                   "nl": "Wie volgt die cursus, en wanneer weet je wie er zakte?"},
        "diploma": {"en": "Who gets a diploma date at all?",
                    "nl": "Wie krijgt er überhaupt een datum van een diploma?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the columns again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de kolommen opnieuw uit."},
        "empty": {"en": "Put each column with the group it belongs to.",
                  "nl": "Zet elke kolom bij de groep waar hij bij hoort."},
        "unfinished": {"en": "Everything you placed so far is right. Place the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe plaatste, klopt. Plaats de rest ook (nog {n})."},
        "wrong": {"en": "{item} goes in the other group.", "nl": "{item} hoort in de andere groep."},
        "correct": {"en": "All {n} are right. The days absent are the sneaky one: a fine feature for a prediction in June, but on 1 September nobody has counted them yet. A model trained with them would look good in the test and be no use on 1 September.",
                    "nl": "Alle {n} kloppen. De dagen afwezig zijn de stiekeme: een prima feature voor een voorspelling in juni, maar op 1 september heeft niemand ze nog geteld. Een model dat ermee getraind is, ziet er goed uit in de test en heb je op 1 september niets aan."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("inside-folds", _F_START, "empty", None),
    ("inside-folds", _F, "correct", None),
    ("inside-folds", _f("select(X[train], y[train], k=20)"), "correct", None),
    ("inside-folds", _F_HEAD + "    X_train, y_train = X[train], y[train]\n    cols = select(X_train, y_train, 20)\n" + _F_TAIL, "correct", None),
    ("inside-folds", _f("select(X, y, 20)"), "wrong", "all-people"),
    ("inside-folds", _f("select(X[val], y[val], 20)"), "wrong", "validation-people"),
    ("inside-folds", _f("select(X[train], y, 20)"), "error", "other-people"),
    ("inside-folds", _f("select(X[train], y[val], 20)"), "error", "other-people"),
    ("inside-folds", _f("select(X[train], y[train], 19)"), "almost", "other-value"),
    ("inside-folds", _f("select(X[train], y[train], 21)"), "almost", "other-value"),
    ("inside-folds", _f("np.arange(20)"), "wrong", None),
    ("inside-folds", _f("select(X[train], y[train], 20)").replace("right / 50\n", "print(right / 50)\n"), "wrong", "print-last"),
    ("known-then", {"grades": "known", "distance": "known", "subjects": "known", "absences": "leaks", "summer": "leaks", "diploma": "leaks"}, "correct", None),
    ("known-then", {"grades": "known", "distance": "known", "subjects": "known", "absences": "known"}, "wrong", "wrong:absences"),
    ("known-then", {"grades": "known"}, "wrong", "unfinished"),
]
