# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-imbalanced: the 10,000 runs of the synthetic AI4I 2020 dataset, 339 of them failures, split
as the worked example splits them (NumPy's generator with seed 0, 3,000 runs to test on and 7,000 to train on), with
the five numbers of every run scaled by the mean and standard deviation of the training part; logistic regression of
scikit-learn trained without weights, with class weights, after undersampling, after oversampling and after SMOTE; the
functions balanced_weights and new_failure. The numbers in capitals were measured once; tests/test_ml_imbalanced.py
measures them again."""
import contextlib
import functools
import io
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/ai4i-2020.csv.gz"
NUMBERS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
ROWS, FAILURES = 10_000, 339
TRAIN, TRAIN_FAILURES, TEST, TEST_FAILURES = 7_000, 229, 3_000, 110
ALWAYS_NO = {"all": 0.966, "test": 0.963}          # the share right of "no failure" every time
PLAIN_ACCURACY = 0.972                              # logistic regression without weights, on the test runs
BALANCED = (15.28, 0.517)                           # the balanced weight of a failure and of a normal run
# Logistic regression on the 3,000 test runs: failures found (of 110), alarms, and AUC.
RESULTS = {"plain": (31, 35, 0.888), "balanced": (89, 630, 0.899), "under": (91, 684, 0.896),
           "over": (89, 629, 0.900), "smote": (89, 622, 0.901), "threshold": (87, 630, 0.888)}
THRESHOLD = 0.031          # the threshold that gives the model without weights as many alarms as the balanced one
NEW = 6_542                # new failures of oversampling and SMOTE: 6,771 - 229
K = 5                      # SMOTE: the number of nearest other failures to choose from
# The slider: the class weight of a failure, a normal run keeping weight 1; failures found and alarms.
WEIGHTS = [1, 2, 5, 10, 30, 100]
WEIGHTED = {1: (31, 35), 2: (43, 63), 5: (63, 142), 10: (72, 274), 30: (89, 636), 100: (102, 1282)}
# Exercise 3: k-NN with k = 5 after oversampling the training part (failures found, alarms), and after oversampling
# all runs before the split: test rows, failures among them, failures found.
KNN_OVER = (83, 179)
LEAK = (5_796, 2_913, 2_913)


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def percent(x: float, lang: str) -> str:
    return number(100 * x, lang, 1) + "%"


@functools.lru_cache(maxsize=1)
def machines():
    """The five numbers of every run, not scaled, and the label: 1 for a failure."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(paths.ROOT / DATA)
    return np.array(df[NUMBERS], dtype=float), np.array(df["Machine failure"])


@functools.lru_cache(maxsize=1)
def split():
    """The positions of the test runs and of the runs to train on, and the numbers scaled with the training part."""
    import numpy as np
    X, y = machines()
    order = np.random.default_rng(0).permutation(len(y))
    test, train = order[:TEST], order[TEST:]
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    return test, train, (X - mean) / std, mean, std


def found_and_alarms(alarm) -> tuple:
    """How many of the test failures an array of alarms for the test runs finds, and how many alarms it raises."""
    import numpy as np
    test, _, _, _, _ = split()
    _, y = machines()
    alarm = np.asarray(alarm, dtype=bool)
    return int(y[test][alarm].sum()), int(alarm.sum())


def fit_and_count(sample_weight=None) -> tuple:
    """Logistic regression on the training part with these weights: failures found and alarms on the test runs."""
    from sklearn.linear_model import LogisticRegression
    test, train, Z, _, _ = split()
    _, y = machines()
    model = LogisticRegression().fit(Z[train], y[train], sample_weight=sample_weight)
    return found_and_alarms(model.predict(Z[test]) == 1)


@functools.lru_cache(maxsize=1)
def plain_curve() -> tuple:
    """The precision and recall of the model without weights at every threshold, from the surest alarm on: lists in %."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    test, train, Z, _, _ = split()
    _, y = machines()
    p = LogisticRegression().fit(Z[train], y[train]).predict_proba(Z[test])[:, 1]
    hits = np.cumsum(y[test][np.argsort(-p, kind="stable")])
    alarms = np.arange(1, len(hits) + 1)
    return (100 * hits / TEST_FAILURES).tolist(), (100 * hits / alarms).tolist()


# --- the slider -----------------------------------------------------------------------------------------------------

def weight_said(w: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    found, alarms = WEIGHTED[w]
    rec, prec = found / TEST_FAILURES, found / alarms
    if lang == "nl":
        start = ("Weight 1, geen extra weight: " if w == 1 else
                 "Weight 30, ongeveer de gebalanceerde weight: " if w == 30 else f"Weight {w}: ")
        return (start + f"{number(alarms, lang)} alarmen, en {found} van de 110 storingen gevonden. "
                f"Recall {percent(rec, lang)}, precision {percent(prec, lang)}.")
    start = ("Weight 1, no extra weight: " if w == 1 else
             "Weight 30, about the balanced weight: " if w == 30 else f"Weight {w}: ")
    return (start + f"{number(alarms, lang)} alarms, and {found} of the 110 failures found. "
            f"Recall {percent(rec, lang)}, precision {percent(prec, lang)}.")


def weight_picture(w: int, labels: dict) -> str:
    """The precision-recall curve of the model without weights, and the models trained with every weight of the
    slider at threshold 0.5, the one of `w` drawn large."""
    rec, prec = plain_curve()
    p = chart.Panel((0, 100), (0, 100), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100])
    p.line(rec, prec, role="ink-3", width=1.4)
    for v in WEIGHTS:
        found, alarms = WEIGHTED[v]
        x, y = 100 * found / TEST_FAILURES, 100 * found / alarms
        if v == w:
            p.points([x], [y], role="ink", r=5.5)
            if x > 70:
                p.text(x - 2.5, y + 6, labels["weight"].format(w=w), role="ink", anchor="end")
            else:
                p.text(x + 2.5, y + 6, labels["weight"].format(w=w), role="ink", anchor="start")
        else:
            p.points([x], [y], role="ink", r=3.2, hollow=True)
    return chart.figure([p], alt=labels["alt"], width=460, height=300)


# The lesson is done when these are correct. "resample-when" is the choice question in the lesson.
REQUIRED = ["balanced-weights", "new-failure", "resample-when"]


class _Val:
    """A test value that the check's messages write by its name, such as `y_train`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _function(seen):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    return fn if callable(fn) else None


def _quiet(work):
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(work)
    except Exception:
        return None


def _call(seen, case: tuple):
    fn = _function(seen)
    return None if fn is None else _quiet(lambda: fn(*(checking._copy(a) for a in case)))


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _array(v, ndim: int):
    """v as a float array with ndim dimensions, or None."""
    import numpy as np
    try:
        a = np.asarray(v, dtype=float)
    except Exception:
        return None
    return a if a.ndim == ndim else None


def _same_array(got, want) -> bool:
    import numpy as np
    a, b = _array(got, np.ndim(want)), _array(want, np.ndim(want))
    return a is not None and b is not None and a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-9, atol=1e-9))


def _shown_array(v):
    import numpy as np
    try:
        a = np.asarray(v, dtype=float)
    except Exception:
        return repr(v)[:60]
    if a.ndim == 0:
        return number(float(a), "en", 3)
    if a.size > 6:
        values = np.unique(np.round(a, 6))
        if a.ndim == 1 and len(values) <= 3:
            return {"en": f"{number(a.size, 'en')} values, each " + " or ".join(_short(x, "en") for x in values),
                    "nl": f"{number(a.size, 'nl')} waarden, elk " + " of ".join(_short(x, "nl") for x in values)}
        return {"en": f"an array of {number(a.size, 'en')} values", "nl": f"een array met {number(a.size, 'nl')} waarden"}
    return "[" + ", ".join(f"{x:.3f}" for x in a.ravel()) + "]"


def _short(x: float, lang: str) -> str:
    """x with at most 3 decimals and no zeros at the end, such as 15.284, 0.517 or 1, and a small x with two
    significant digits, such as 0.0044."""
    text = f"{x:.2g}" if 0 < abs(x) < 0.1 else f"{x:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


# --- exercise 1: balanced weights ------------------------------------------------------------------------------------

def _labels(name: str, values: list):
    import numpy as np
    return (_Val(name, np.array(values)),)


_Y_TRAIN = (_Val("y_train", machines()[1][split()[1]]),)
_Y1 = _labels("y_small", [0, 0, 0, 1])
_Y2 = _labels("y_half", [1, 0, 1, 0, 0, 1])
_Y3 = _labels("y_rare", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

_W_HEAD = "import numpy as np\n\n\ndef balanced_weights(y):\n    n = len(y)\n    failures = y.sum()\n    weights = np.zeros(n)\n"


def _w(fail: str = "n / (2 * failures)", normal: str = "n / (2 * (n - failures))") -> str:
    return _W_HEAD + f"    weights[y == 1] = {fail}\n    weights[y == 0] = {normal}\n    return weights\n"


_W_START = _w("...", "...")
_W = _w()

_BW_WORDS = {
    "title": {"en": "Your weights: how much each class weighs in total", "nl": "Jouw weights: hoeveel elke klasse in totaal weegt"},
    "normal": {"en": "normal runs", "nl": "normale runs"},
    "failures": {"en": "failures", "nl": "storingen"},
    "count": {"en": "count", "nl": "aantal"},
    "total": {"en": "total weight", "nl": "totale weight"},
    "right": {"en": "With your weights both classes weigh 3,500 in total: 229 failures of 15.28 and 6,771 normal runs of 0.517. Logistic regression trained with them finds 89 of the 110 test failures, with 630 alarms: recall 80.9%, precision 14.1%. Without weights it finds 31, with 35 alarms.",
              "nl": "Met jouw weights wegen beide klassen in totaal 3.500: 229 storingen van 15,28 en 6.771 normale runs van 0,517. Logistische regressie die daarmee traint, vindt 89 van de 110 storingen in de test set, met 630 alarmen: recall 80,9%, precision 14,1%. Zonder weights vindt het model er 31, met 35 alarmen."},
    "said": {"en": "With your weights the failures weigh {fail} in total and the normal runs {normal}. Logistic regression trained with them finds {found} of the 110 test failures, with {alarms} alarms. With balanced weights both weigh 3,500, and it finds 89 with 630 alarms.",
             "nl": "Met jouw weights wegen de storingen in totaal {fail} en de normale runs {normal}. Logistische regressie die daarmee traint, vindt {found} van de 110 storingen in de test set, met {alarms} alarmen. Met gebalanceerde weights wegen beide 3.500, en vindt het model er 89 met 630 alarmen."},
    "unfit": {"en": "With your weights the failures weigh {fail} in total and the normal runs {normal}. With balanced weights both weigh 3,500.",
              "nl": "Met jouw weights wegen de storingen in totaal {fail} en de normale runs {normal}. Met gebalanceerde weights wegen beide 3.500."},
}


def _bw_picture(seen) -> str:
    import numpy as np
    lang = seen.lang
    weights = _array(_call(seen, _Y_TRAIN), 1)
    y = _Y_TRAIN[0].value
    if weights is None or len(weights) != len(y) or not np.all(np.isfinite(weights)):
        return ""
    words = {k: v[lang] for k, v in _BW_WORDS.items()}
    fail, normal = float(weights[y == 1].sum()), float(weights[y == 0].sum())
    counts = [TRAIN - TRAIN_FAILURES, TRAIN_FAILURES]
    totals = [normal, fail]
    hi = max(7_000, max(abs(v) for v in totals)) * 1.25
    lo = min(0.0, min(totals) * 1.12)
    ticks = [t for t in chart.ticks(lo, hi, 4) if lo <= t <= hi]
    p = chart.Panel((lo, hi), (0, 4.2), xlabel=words["total"] + " / " + words["count"], xticks=ticks, yticks=[])
    for row, (name, c, s) in enumerate(zip([words["normal"], words["failures"]], counts, totals)):
        base = 2.2 if row == 0 else 0.1
        p.rect(0, base + 0.95, c, base + 1.45, role="ink-3")
        p.rect(min(0, s), base + 0.35, max(0, s), base + 0.85, role="hyper", solid=True)
        p.text(lo + 0.01 * (hi - lo), base + 1.75, name, role="ink-2", anchor="start")
        p.text(max(c, 0) + 0.015 * (hi - lo), base + 1.2, number(c, lang), role="ink-3", anchor="start")
        p.text(max(s, 0) + 0.015 * (hi - lo), base + 0.6, number(s, lang, 0 if abs(s) >= 100 else 2), role="ink", anchor="start")
    if np.allclose(weights, _W_VALUES):
        text = words["right"]
    elif np.all(weights >= 0) and weights.sum() > 0:
        counted = _quiet(lambda: fit_and_count(weights))
        text = (words["said"].format(fail=number(fail, lang, 1), normal=number(normal, lang, 1), found=counted[0],
                                     alarms=number(counted[1], lang))
                if counted else words["unfit"].format(fail=number(fail, lang, 1), normal=number(normal, lang, 1)))
    else:
        text = words["unfit"].format(fail=number(fail, lang, 1), normal=number(normal, lang, 1))
    key = (f'<span style="color:var(--ea-ink-3)">▭</span> {words["count"]} &nbsp; '
           f'<span class="c-hyper">■</span> {words["total"]}')
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=230) + _key(key) + _key(text), alt=text)


def _course_weights(y):
    import numpy as np
    n, failures = len(y), y.sum()
    weights = np.zeros(n)
    weights[y == 1] = n / (2 * failures)
    weights[y == 0] = n / (2 * (n - failures))
    return weights


_W_VALUES = _course_weights(_Y_TRAIN[0].value)

EXERCISES = {
    "balanced-weights": FunctionBox(
        name="balanced-weights",
        picture=_bw_picture,
        function="balanced_weights",
        start=_W_START,
        solution=_W,
        show=_shown_array,
        compare=_same_array,
        visible=[_Y_TRAIN],
        hidden=[_Y1, _Y2, _Y3],
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_w("n / (2 * (n - failures))", "n / (2 * failures)"),
                    en="`{call}` gives {got}: every failure gets the small weight and every normal run the large one, so the normal runs count even more than they already did. The class with fewer examples gets the larger weight: divide by the number of failures for a failure.",
                    nl="`{call}` geeft {got}: elke storing krijgt de kleine weight en elke normale run de grote, dus de normale runs tellen nog meer dan ze al deden. De klasse met minder voorbeelden krijgt de grotere weight: deel voor een storing door het aantal storingen."),
            Mistake("share", wrong=_w("failures / n", "(n - failures) / n"),
                    en="`{call}` gives {got}: the share of each class, 0.033 for a failure. That makes the rare class count less. The weight is the other way round: n / 2 shared by the examples of the class, so the fewer there are, the more each weighs.",
                    nl="`{call}` geeft {got}: het aandeel van elke klasse, 0,033 voor een storing. Daarmee telt de zeldzame klasse minder. De weight is precies andersom: n / 2 verdeeld over de voorbeelden van de klasse, dus hoe minder er zijn, hoe meer elk weegt."),
            Mistake("no-two", wrong=_w("n / failures", "n / (n - failures)"),
                    en="`{call}` gives {got}: the two classes weigh the same, so the ratio is right, but each weighs n and together 2n. With the 2, each class weighs n / 2, and all weights together add up to n, as many as there are examples: the loss keeps its size. Divide by 2 times the count.",
                    nl="`{call}` geeft {got}: de twee klassen wegen even zwaar, dus de verhouding klopt, maar elk weegt n en samen 2n. Met de 2 weegt elke klasse n / 2, en tellen alle weights samen op tot n, evenveel als er voorbeelden zijn: de loss houdt zijn grootte. Deel door 2 keer het aantal."),
            Mistake("normal-one", wrong=_w(normal="1"),
                    en="`{call}` gives {got}: every failure weighs right, but a normal run keeps weight 1. Then the 229 failures weigh 3,500 in total and the 6,771 normal runs 6,771: the classes are not balanced. A normal run gets its share of n / 2 as well.",
                    nl="`{call}` geeft {got}: elke storing weegt goed, maar een normale run houdt weight 1. Dan wegen de 229 storingen samen 3.500 en de 6.771 normale runs 6.771: de klassen zijn niet in evenwicht. Ook een normale run krijgt zijn deel van n / 2."),
            Mistake("ratio", wrong=_w("(n - failures) / failures", "1"),
                    en="`{call}` gives {got}: a failure weighs 29.6 times as much as a normal run, the right ratio, but together the classes weigh 13,542 instead of n = 7,000. Balanced weights let each class weigh n / 2 in total, so that the loss keeps its size.",
                    nl="`{call}` geeft {got}: een storing weegt 29,6 keer zoveel als een normale run, de goede verhouding, maar samen wegen de klassen 13.542 in plaats van n = 7.000. Gebalanceerde weights laten elke klasse in totaal n / 2 wegen, zodat de loss zijn grootte houdt."),
            Mistake("one-each", wrong=_w("1 / failures", "1 / (n - failures)"),
                    en="`{call}` gives {got}: the two classes weigh the same, so the ratio is right, but each class weighs 1 in total, and all 7,000 weights together 2. Balanced weights add up to n, as many as there are examples, so that the loss keeps its size: n / (2 times the count).",
                    nl="`{call}` geeft {got}: de twee klassen wegen even zwaar, dus de verhouding klopt, maar elke klasse weegt in totaal 1, en alle 7.000 weights samen 2. Gebalanceerde weights tellen op tot n, evenveel als er voorbeelden zijn, zodat de loss zijn grootte houdt: n / (2 keer het aantal)."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the weight of a failure and the weight of a normal run.",
                      "nl": "Vervang de twee `...`: de weight van een storing en de weight van een normale run."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(y)`.",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: n / (2 times the number of failures) for a failure, n / (2 times the number of normal runs) for a normal run.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: n / (2 keer het aantal storingen) voor een storing, n / (2 keer het aantal normale runs) voor een normale run."},
            "almost": {"en": "Right for the 7,000 runs. But `{call}` gives {got}, and it should give {want}. Does your function count the failures of the `y` it gets, and use no fixed number such as 229 or 7,000?",
                       "nl": "Goed voor de 7.000 runs. Maar `{call}` geeft {got}, en dat moet {want} zijn. Telt je functie de storingen van de `y` die hij krijgt, en niets van de machines?"},
            "correct": {"en": "Your function is right in all {n} cases. Every failure weighs 15.28 and every normal run 0.517; scikit-learn's `class_weight=\"balanced\"` gives the same weights.",
                        "nl": "Je functie klopt in alle {n} gevallen. Elke storing weegt 15,28 en elke normale run 0,517; `class_weight=\"balanced\"` van scikit-learn geeft dezelfde weights."},
        },
    ),
}


# --- exercise 2: a new failure with SMOTE ----------------------------------------------------------------------------

def _rare():
    test, train, Z, _, _ = split()
    _, y = machines()
    return Z[train][y[train] == 1]


_FAILS = _Val("failures", _rare())


def _points(name: str, values: list):
    import numpy as np
    return _Val(name, np.array(values, dtype=float))


# Points whose distances never tie: from (0, 0) the nearest other is (1, 0), then (0, 2), (-1, -2) and (3, 3).
_P = _points("points", [[0, 0], [1, 0], [0, 2], [3, 3], [-1, -2]])
_N_VISIBLE = (_FAILS, 0, 2, 0.4)
_N_HIDDEN = [(_P, 0, 0, 0.5), (_P, 3, 1, 0.25), (_P, 0, 2, 1.0), (_P, 4, 0, 0.0), (_P, 2, 3, 0.75)]

_N_HEAD = ("import numpy as np\n\n\ndef new_failure(X_rare, i, j, u):\n"
           "    distances = np.linalg.norm(X_rare - X_rare[i], axis=1)\n")


def _n(nearest: str = "np.argsort(distances)[1:]", back: str = "X_rare[i] + u * (b - X_rare[i])") -> str:
    return (_N_HEAD + f"    nearest = {nearest}\n    b = X_rare[nearest[j]]\n    return {back}\n")


_N_START = _N_HEAD + "    nearest = ...\n    b = X_rare[nearest[j]]\n    return ...\n"
_N = _n()


def _course_new(X_rare, i, j, u):
    import numpy as np
    distances = np.linalg.norm(X_rare - X_rare[i], axis=1)
    nearest = np.argsort(distances)[1:]
    b = X_rare[nearest[j]]
    return X_rare[i] + u * (b - X_rare[i])


def _draws(count: int, seed: int = 3) -> list:
    """The i, j and u of the course's SMOTE, as the lesson draws them: NumPy's generator with seed 3."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(_FAILS.value)
    return [(int(rng.integers(n)), int(rng.integers(K)), float(rng.random())) for _ in range(count)]


_NF_WORDS = {
    "title": {"en": "Your new failures, in two of the five numbers", "nl": "Jouw nieuwe storingen, in twee van de vijf getallen"},
    "x": {"en": "rotational speed (rpm)", "nl": "toerental (rpm)"},
    "y": {"en": "torque (Nm)", "nl": "koppel (Nm)"},
    "normal": {"en": "normal runs", "nl": "normale runs"},
    "failures": {"en": "failures", "nl": "storingen"},
    "new": {"en": "your new failures", "nl": "jouw nieuwe storingen"},
    "right": {"en": "The filled dots are the 229 failures of the training part, the open ones 229 new failures that your function made, each between a failure and one of its 5 nearest other failures, in all five numbers. They stay where the failures are: most at a low speed with a high torque, some at a high speed with a low torque. Copies would sit exactly on the dots.",
              "nl": "De dichte stippen zijn de 229 storingen uit het trainingsdeel, de open stippen 229 nieuwe storingen die jouw functie maakte, elk tussen een storing en een van zijn 5 naaste andere storingen, in alle vijf de getallen. Ze blijven waar de storingen zijn: de meeste bij een laag toerental met een hoog koppel, een paar bij een hoog toerental met een laag koppel. Kopieën zouden precies op de stippen liggen."},
    "said": {"en": "The filled dots are the 229 failures of the training part, the open ones the 229 points that your function made from them, with the same i, j and u as the course's SMOTE. {where}",
             "nl": "De dichte stippen zijn de 229 storingen uit het trainingsdeel, de open stippen de 229 punten die jouw functie ervan maakte, met dezelfde i, j en u als de SMOTE van de course. {where}"},
    "off": {"en": "{n} of them differ from what the course's function makes with the same i, j and u.",
            "nl": "{n} ervan verschillen van wat de functie van de course maakt met dezelfde i, j en u."},
    "off-1": {"en": "1 of them differs from what the course's function makes with the same i, j and u.",
              "nl": "1 ervan verschilt van wat de functie van de course maakt met dezelfde i, j en u."},
    "copies": {"en": "{n} of them are a copy of a failure.", "nl": "{n} ervan zijn een kopie van een storing."},
    "copies-1": {"en": "1 of them is a copy of a failure.", "nl": "1 ervan is een kopie van een storing."},
}


def _nf_picture(seen) -> str:
    import numpy as np
    lang = seen.lang
    fn = _function(seen)
    if fn is None:
        return ""
    R = _FAILS.value
    draws = _draws(len(R))
    made = _quiet(lambda: [fn(checking._copy(R), i, j, u) for i, j, u in draws])
    if made is None:
        return ""
    new = _array(made, 2)
    if new is None or new.shape != R.shape or not np.all(np.isfinite(new)):
        return ""
    words = {k: v[lang] for k, v in _NF_WORDS.items()}
    course = np.array([_course_new(R, i, j, u) for i, j, u in draws])
    if np.allclose(new, course):
        text = words["right"]
    else:
        off = sum(not np.allclose(a, b) for a, b in zip(new, course))
        copies = sum(bool(np.any(np.all(np.isclose(R, a), axis=1))) for a in new)
        said_copies = words["copies-1"] if copies == 1 else words["copies"].format(n=copies)
        said_off = words["off-1"] if off == 1 else words["off"].format(n=off)
        where = (said_copies + " " if copies else "") + said_off
        text = words["said"].format(where=where)
    _, train, Z, mean, std = split()
    _, y = machines()
    normal = Z[train][y[train] == 0][::7]
    unscaled = lambda A: (A[:, 2] * std[2] + mean[2], A[:, 3] * std[3] + mean[3])
    p = chart.Panel((1100, 2950), (0, 80), xlabel=words["x"], ylabel=words["y"], xticks=[1200, 1600, 2000, 2400, 2800],
                    yticks=[0, 20, 40, 60, 80])
    x, yv = unscaled(normal)
    p.points(x, yv, role="axis", r=1.1)
    x, yv = unscaled(R)
    p.points(x, yv, role="data", r=3)
    x, yv = unscaled(new)
    keep = (x >= 1100) & (x <= 2950) & (yv >= 0) & (yv <= 80)
    p.points(x[keep], yv[keep], role="data", r=2.8, hollow=True)
    key = (f'<span style="color:var(--ea-axis)">·</span> {words["normal"]} &nbsp; '
           f'<span style="color:var(--ea-data)">●</span> {words["failures"]} &nbsp; '
           f'<span style="color:var(--ea-data)">○</span> {words["new"]}')
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=300) + _key(key) + _key(text), alt=text)


EXERCISES["new-failure"] = FunctionBox(
    name="new-failure",
    picture=_nf_picture,
    function="new_failure",
    start=_N_START,
    solution=_N,
    show=_shown_array,
    compare=_same_array,
    visible=[_N_VISIBLE],
    hidden=_N_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("itself", wrong=_n("np.argsort(distances)"),
                en="`{call}` gives {got}. The nearest failure of all is failure `i` itself, at distance 0, so `nearest[0]` is `i`, and with `j = 0` you get a copy of it. Leave the first position out.",
                nl="`{call}` geeft {got}. De naaste storing van allemaal is storing `i` zelf, op afstand 0, dus `nearest[0]` is `i`, en met `j = 0` krijg je een kopie ervan. Laat de eerste plek weg."),
        Mistake("from-b", wrong=_n(back="b + u * (X_rare[i] - b)"),
                en="`{call}` gives {got}: a point on the right line, but measured from `b`, so it lies at share `1 - u` from failure `i`. Start at `X_rare[i]` and go a share `u` toward `b`.",
                nl="`{call}` geeft {got}: een punt op de goede lijn, maar gemeten vanaf `b`, dus het ligt op aandeel `1 - u` vanaf storing `i`. Begin bij `X_rare[i]` en ga een aandeel `u` richting `b`."),
        Mistake("middle", wrong=_n(back="(X_rare[i] + b) / 2"),
                en="`{call}` gives {got}: always the middle, whatever `u` is. SMOTE puts every new failure at a random share `u` of the way: start at `X_rare[i]` and go a share `u` of the step toward `b`.",
                nl="`{call}` geeft {got}: altijd het midden, wat `u` ook is. SMOTE zet elke nieuwe storing op een willekeurig aandeel `u` van de weg: begin bij `X_rare[i]` en zet een aandeel `u` van de stap richting `b`."),
        Mistake("no-start", wrong=_n(back="u * (b - X_rare[i])"),
                en="`{call}` gives {got}: only the step, `u` times the way from `X_rare[i]` to `b`, near zero. The new failure is that step taken from failure `i`: add `X_rare[i]`.",
                nl="`{call}` geeft {got}: alleen de stap, `u` keer de weg van `X_rare[i]` naar `b`, dicht bij nul. De nieuwe storing is die stap gezet vanaf storing `i`: tel `X_rare[i]` erbij op."),
        Mistake("far", wrong=_n("np.argsort(distances)[::-1]"),
                en="`{call}` gives {got}: `[::-1]` turns the order around, so `nearest[0]` is the failure farthest away. `np.argsort` already starts at the nearest; leave out only the first position, failure `i` itself.",
                nl="`{call}` geeft {got}: `[::-1]` draait de volgorde om, dus `nearest[0]` is de storing die het verst weg ligt. `np.argsort` begint al bij de naaste; laat alleen de eerste plek weg, storing `i` zelf."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the positions of the other failures, nearest first, and the point at share `u` from failure `i` toward `b`.",
                  "nl": "Vervang de twee `...`: de plekken van de andere storingen, de naaste eerst, en het punt op aandeel `u` vanaf storing `i` richting `b`."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X_rare, i, j, u)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X_rare, i, j, u)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the point at share `u` on the way from failure `i` to its nearest other failure number `j`.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: het punt op aandeel `u` van de weg van storing `i` naar zijn naaste andere storing nummer `j`."},
        "almost": {"en": "Right for the failures. But `{call}` gives {got}, and it should give {want}. Does your function use its four values, and no fixed number such as 229?",
                   "nl": "Goed voor de storingen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie zijn vier waarden, en geen vast getal zoals 229?"},
        "correct": {"en": "Your function is right in all {n} cases. Drawn 6,542 times with a random `i`, `j` and `u`, it gives the SMOTE failures of the table.",
                    "nl": "Je functie klopt in alle {n} gevallen. 6.542 keer getrokken met een willekeurige `i`, `j` en `u` geeft hij de SMOTE-storingen uit de tabel."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("balanced-weights", _W_START, "empty", None),
    ("balanced-weights", _w(normal="..."), "wrong", "dots-left"),
    ("balanced-weights", _W, "correct", None),
    ("balanced-weights", _w("len(y) / (2 * y.sum())", "len(y) / (2 * (len(y) - y.sum()))"), "correct", None),
    ("balanced-weights", _w("n / 2 / failures", "n / 2 / (y == 0).sum()"), "correct", None),
    ("balanced-weights", _w("n / (2 * (n - failures))", "n / (2 * failures)"), "wrong", "swapped"),
    ("balanced-weights", _w("failures / n", "(n - failures) / n"), "wrong", "share"),
    ("balanced-weights", _w("n / failures", "n / (n - failures)"), "wrong", "no-two"),
    ("balanced-weights", _w("1 / failures", "1 / (n - failures)"), "wrong", "one-each"),
    ("balanced-weights", _w(normal="1"), "wrong", "normal-one"),
    ("balanced-weights", _w("(n - failures) / failures", "1"), "wrong", "ratio"),
    ("balanced-weights", _w("n // (2 * failures)", "n // (2 * (n - failures))"), "wrong", None),
    ("new-failure", _N_START, "empty", None),
    ("new-failure", _n(back="..."), "wrong", "dots-left"),
    ("new-failure", _N, "correct", None),
    ("new-failure", _n(back="(1 - u) * X_rare[i] + u * b"), "correct", None),
    ("new-failure", _n("np.argsort(distances)[1:6]"), "correct", None),
    ("new-failure", _n("np.argsort(distances)"), "wrong", "itself"),
    ("new-failure", _n(back="b + u * (X_rare[i] - b)"), "wrong", "from-b"),
    ("new-failure", _n(back="(X_rare[i] + b) / 2"), "wrong", "middle"),
    ("new-failure", _n(back="u * (b - X_rare[i])"), "wrong", "no-start"),
    ("new-failure", _n("np.argsort(distances)[::-1]"), "wrong", "far"),
]
