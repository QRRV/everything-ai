# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-project-audit-explain, the second step of the project audit: three functions the learner
writes and keeps in the project, importance (how much less often the classifier is right when one column is
shuffled), dependence (the mean probability of a yes when everyone gets one value in a column) and fewest_hours (the
fewest hours a week that give one person a yes); with the pictures of what they make of the course's classifier, and
what the learner's project makes of the learner's own classifier. The Shapley values of a logistic regression's logit
are each column's weights times how far the person lies from the average of the people to train on (shapley_logit)."""
import html
import importlib.util
import math
import numbers
import re
import warnings

import numpy as np
import pandas as pd

from everything_ai import chart, paths, projects, yours
from everything_ai import check as checking
from everything_ai.check import FunctionBox, Mistake

# The report of step one: this lesson uses the learner's report, or the course's version when it is missing.
projects.builds_on(("audit", "report", "rai-project-audit-bias", "report"))

# What the course's classifier gives (tests/test_rai_project_audit_explain.py checks every number).
RIGHT = 0.834               # how often it is right on the test set, with one threshold of 0.5
PERSON = 34982              # the row of the worked example: a woman aged 40 with a no, never married
SEED_COUNT = 3              # the seeds 0, 1 and 2 of the demo's permutation importance
SAMPLE = 100                # the demo's people with a no per sex, drawn with random_state 0
TOP = 99                    # the most hours a week anyone in the data works


def _course_model():
    """The course's own audit/model.py, as the lesson gives it to the learner: not the learner's copy."""
    spec = importlib.util.spec_from_file_location(
        "everything_ai_course_audit_model_2",
        paths.LESSONS / "rai-project-audit-explain" / "project" / "audit" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- scores_of: the classifier's probability of a yes for any table, also a table of one row ------------------------

def _parts_of(module, people: pd.DataFrame) -> tuple:
    """What a scorer needs from a model.py: the trained model, its columns, and the whole table's maxima."""
    numbers_, words = list(module.NUMBERS), list(module.WORDS)
    model = module.train(people)
    columns = list(module.features(people).columns)
    top = pd.get_dummies(people[numbers_ + words], columns=words, dtype=float).max()
    return model, numbers_, words, columns, top


def _matrix(t: pd.DataFrame, numbers_: list, words: list, columns: list, top) -> np.ndarray:
    """The columns features() makes from the whole table, for the rows of t: numbers divided by their maximum in the
    whole table, and a column of 0 and 1 per value of a word column, as pd.get_dummies makes them."""
    pos = {c: i for i, c in enumerate(columns)}
    X = np.zeros((len(t), len(columns)))
    rows = np.arange(len(t))
    for c in numbers_:
        X[:, pos[c]] = t[c].to_numpy(dtype=float) / float(top[c])
    for c in words:
        mine = {name[len(c) + 1:]: i for name, i in pos.items() if name.startswith(c + "_")}
        where = np.array([mine.get(v, -1) for v in t[c].tolist()], dtype=int)
        ok = where >= 0
        X[rows[ok], where[ok]] = 1.0
    return X


class Scorer:
    """scores_of(table): the classifier's probability of a yes for every row of the table, in order, as an array. It
    makes the classifier's columns from the whole table's values, so that it works on a table of one row too, where
    features() alone would see only that row. The check calls it by its name, and the learner's code gets this one."""

    def __init__(self, make, name: str = "scores_of"):
        self.make, self.name, self._parts = make, name, None

    def parts(self) -> tuple:
        if self._parts is None:
            self._parts = self.make()
        return self._parts

    def __call__(self, t):
        model, numbers_, words, columns, top = self.parts()
        with warnings.catch_warnings():   # a model fitted on a table, asked about an array
            warnings.simplefilter("ignore")
            return model.predict_proba(_matrix(t, numbers_, words, columns, top))[:, 1]

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


class Fake:
    """A made-up classifier for the check's own small cases: a function of a table, by its name."""

    def __init__(self, fn, name: str):
        self.fn, self.name = fn, name

    def __call__(self, t):
        return np.asarray(self.fn(t), dtype=float)

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


_DATA: dict = {}


def table() -> pd.DataFrame:
    """UCI Adult with rich: 1 when the person earns more than 50,000 dollars, else 0. Read-only."""
    if "people" not in _DATA:
        people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
        people["rich"] = (people["income"] == ">50K").astype(int)
        _DATA["people"] = people
    return _DATA["people"]


def _course_parts() -> tuple:
    if "parts" not in _DATA:
        _DATA["parts"] = _parts_of(_course_model(), table())
    return _DATA["parts"]


SCORES = Scorer(_course_parts)


def test_part() -> pd.DataFrame:
    people = table()
    return people[people["split"] == "test"]


def train_part() -> pd.DataFrame:
    people = table()
    return people[people["split"] == "train"]


class Table:
    """A table the check calls by the name the lesson gives it, such as PEOPLE, so that a message stays short. It is
    made the first time a check needs it, and the learner's code gets its own copy."""

    def __init__(self, make, name: str):
        self.make, self.name, self._df = make, name, None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self.make()
        return self._df

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.df.copy()


PEOPLE = Table(test_part, "PEOPLE")
WOMEN = Table(lambda: test_part()[test_part()["sex"] == "Female"], "WOMEN")
MEN = Table(lambda: test_part()[test_part()["sex"] == "Male"], "MEN")
ONE = Table(lambda: table().loc[[PERSON]], "ONE")
# a married man with a yes at 40 hours, who would get a yes from 25 hours a week on
MARRIED = Table(lambda: table().loc[[32572]], "MARRIED")

# six made-up people: the classifier BY_A says yes from a = 4, and is right about all six
SIX = Table(lambda: pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [6, 5, 4, 3, 2, 1],
                                  "rich": [0, 0, 0, 1, 1, 1]}), "SIX")
BY_A = Fake(lambda t: t["a"].to_numpy() / 7, "by_a")
BY_B = Fake(lambda t: t["b"].to_numpy() / 7, "by_b")
# one made-up person, and classifiers of the hours alone: by_hours gives exactly 0.5 at 50 hours
ROW = Table(lambda: pd.DataFrame({"age": [30], "hours_per_week": [40], "rich": [0]}), "ROW")
BY_HOURS = Fake(lambda t: t["hours_per_week"].to_numpy() / 100, "by_hours")
FEW = Fake(lambda t: np.where(t["hours_per_week"].to_numpy() <= 20, 0.9, 0.1), "few_hours")


# --- the course's own functions, as the lesson writes them -----------------------------------------------------------

def importance(scores_of, people, column, seed):
    shuffled = people.copy()
    shuffled[column] = people[column].sample(frac=1, random_state=seed).to_numpy()
    right = ((scores_of(people) > 0.5) == (people["rich"] == 1)).mean()
    after = ((scores_of(shuffled) > 0.5) == (shuffled["rich"] == 1)).mean()
    return right - after


def dependence(scores_of, people, column, value):
    changed = people.copy()
    changed[column] = value
    return scores_of(changed).mean()


def fewest_hours(scores_of, person, top):
    changed = person.copy()
    for hours in range(1, top + 1):
        changed["hours_per_week"] = hours
        if scores_of(changed)[0] > 0.5:
            return hours
    return None


def shapley_logit(parts, rows: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame | None:
    """The Shapley values of a logistic regression's logit, per column of the table, for every row: each of the
    column's weights times how far the row lies from the background's average, added over the 0/1 columns of a word
    column. None for a model without one weight per column."""
    model, numbers_, words, columns, top = parts
    coef = getattr(model, "coef_", None)
    if coef is None or np.shape(coef) != (1, len(columns)):
        return None
    w = np.asarray(coef[0], dtype=float)
    X, B = _matrix(rows, numbers_, words, columns, top), _matrix(background, numbers_, words, columns, top)
    shares = (X - B.mean(axis=0)) * w
    out = {}
    for c in numbers_ + words:
        mine = [i for i, name in enumerate(columns) if name == c or name.startswith(c + "_")]
        out[c] = shares[:, mine].sum(axis=1)
    return pd.DataFrame(out, index=rows.index)


def base_logit(parts, background: pd.DataFrame) -> float:
    """The logit of the background's average row: what the Shapley values of a row add up from."""
    model, numbers_, words, columns, top = parts
    B = _matrix(background, numbers_, words, columns, top)
    return float(model.intercept_[0] + B.mean(axis=0) @ np.asarray(model.coef_[0], dtype=float))


# --- numbers in the lesson's language --------------------------------------------------------------------------------

def number(x: float, lang: str, digits: int = 3, sign: bool = False) -> str:
    """A number with this many decimals and a real minus sign, with a plus sign if asked; a decimal comma in Dutch."""
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-+")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if hasattr(x, "item") and getattr(x, "size", 1) == 1 and not isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.item()
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"
COLUMNS = ["age", "education_num", "hours_per_week", "workclass", "marital_status", "occupation", "relationship",
           "race"]
MARITAL = ["Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married",
           "Separated", "Widowed"]


def _mine(seen):
    """The learner's function, or None when the code stopped, the name is not a function, or a `...` of the start is
    still in the code (its check says so first)."""
    if isinstance(seen.text, str) and re.search(r"^[^#\n]*\.\.\.", seen.text, re.MULTILINE):
        return None
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _run(run, seconds: float = 20):
    try:
        return checking.with_time_limit(run, seconds)
    except Exception:   # also Timeout
        return None


# --- pictures: bars per column, dots per value ----------------------------------------------------------------------

def bars(values: dict, words: dict, lang: str, role: str, width: int = 460) -> str:
    """A bar per column with its value, the largest filled; `words` gives "x", "alt" and the row names (`names`)."""
    keys = list(values)
    got = [values[k] for k in keys]
    lo, hi = min(got + [0.0]), max(got + [0.0])
    pad = (hi - lo) * 0.04 + 1e-9
    n = len(keys)
    best = max(range(n), key=lambda i: got[i])
    p = chart.Panel(x=(lo - pad, hi + pad), y=(-0.2, n + 0.2), xlabel=words["x"], ylabel="", yticks=[])
    for i, k in enumerate(keys):
        y = n - 1 - i
        x0, x1 = (0.0, got[i]) if got[i] >= 0 else (got[i], 0.0)
        p.rect(x0, y + 0.04, max(x1, x0 + (hi - lo) * 0.004), y + 0.32, role=role, solid=i == best)
        p.text(lo - pad, y + 0.74, f"{words.get('names', {}).get(k, k)}: {shown(got[i], lang)}", role="ink",
               anchor="start", halo=True)
    svg = chart.figure([p], alt=words["alt"], width=width, height=48 + 40 * n)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def two_groups(rows: list, women: list, men: list, words: dict, lang: str) -> str:
    """A row per value with two dots, open for women and filled for men; `words` gives "x" and "alt"."""
    n = len(rows)
    p = chart.Panel(x=(0, 1), y=(-0.2, n + 0.2), xlabel=words["x"], ylabel="", yticks=[],
                    xticks=[0, 0.25, 0.5, 0.75, 1], xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    for i, name in enumerate(rows):
        y = n - 1 - i + 0.3
        p.text(0, y + 0.42, name, role="ink", anchor="start", halo=True)
        if women[i] is not None:
            p.points([min(max(women[i], 0), 1)], [y], role="data", r=5, hollow=True)
        if men[i] is not None:
            p.points([min(max(men[i], 0), 1)], [y], role="data", r=5)
    svg = chart.figure([p], alt=words["alt"], width=460, height=40 + 38 * n)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- the picture of exercise 1: your importance for every column ---------------------------------------------------

def shown(x: float, lang: str) -> str:
    """A value of a picture: three decimals, or none from 100 on, such as a count instead of a share."""
    return number(x, lang, 0 if abs(x) >= 100 or (abs(x) >= 1 and float(x).is_integer()) else 3)


def _same_as(got, want) -> bool:
    """The learner's values are the course's: the picture may then say what they mean."""
    if isinstance(got, dict):
        return got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-9 for k in got)
    return all(_same_as(a, b) if isinstance(a, list) else abs(a - b) < 1e-9 for a, b in zip(got, want))


def _course(key: str, make):
    if key not in _DATA:
        _DATA[key] = make()
    return _DATA[key]


_IMP_WORDS = {
    "en": {"title": "Your importance(), for every column of the classifier, with seed 0",
           "x": "what your importance() gives on PEOPLE",
           "key": "With your importance on PEOPLE, the largest is {best} ({bv}), and race gets {race}.",
           "flat": "With your importance on PEOPLE, every column gets {bv}.",
           "means": " That is how much less often the classifier is right with the column shuffled: shuffle race, and "
                    "it is right about as often as before.",
           "alt": "A bar per column: what your importance gives on the test set with seed 0. {key} {all}"},
    "nl": {"title": "Jouw importance(), voor elke kolom van de classifier, met seed 0",
           "x": "wat jouw importance() geeft op PEOPLE",
           "key": "Met jouw importance op PEOPLE is de grootste {best} ({bv}), en race krijgt {race}.",
           "flat": "Met jouw importance op PEOPLE krijgt elke kolom {bv}.",
           "means": " Zo veel minder vaak zit de classifier goed met de kolom geschud: schud race, en hij zit ongeveer "
                    "even vaak goed als ervoor.",
           "alt": "Een staaf per kolom: wat jouw importance geeft op de test set met seed 0. {key} {all}"},
}


def importances(fn, scores_of, people, seeds) -> dict | None:
    """What `fn`, an importance function, gives for every column, as the mean over the seeds; None when it fails."""
    out = {}
    for c in COLUMNS:
        got = []
        for seed in seeds:
            v = _number(fn(scores_of, people.copy(), c, seed))
            if v is None:
                return None
            got.append(v)
        out[c] = sum(got) / len(got)
    return out


def _importance_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    SCORES.parts()
    got = _run(lambda: importances(fn, SCORES, PEOPLE.df, [0]))
    if got is None:
        return ""
    lang = seen.lang
    w = dict(_IMP_WORDS[lang])
    best = max(got, key=got.get)
    if max(got.values()) == min(got.values()):
        key = w["flat"].format(bv=shown(got[best], lang))
    else:
        key = w["key"].format(best=best, bv=shown(got[best], lang), race=shown(got["race"], lang))
    if _same_as(got, _course("importances", lambda: importances(importance, SCORES, PEOPLE.df, [0]))):
        key += w["means"]
    every = ", ".join(f"{c} {shown(v, lang)}" for c, v in got.items())
    w["alt"] = w["alt"].format(key=key, all=every)
    svg = bars(got, w, lang, role="loss")
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- the picture of exercise 2: your dependence per value of marital_status, for women and men -----------------------

_DEP_WORDS = {
    "en": {"title": "Your dependence(), for every value of marital_status",
           "x": "what your dependence() gives",
           "key": "Open dots: what your dependence gives for WOMEN, filled: for MEN, at each value of marital_status. "
                  "With Married-civ-spouse it gives {w} for WOMEN and {m} for MEN.",
           "edge": " A value below 0 or above 1 is drawn at the edge.",
           "means": " That is the mean probability of a yes if every woman, or every man, had that value; as they are, "
                    "{w0} and {m0}.",
           "alt": "Two dots per value of marital_status, from your dependence: women {women}; men {men}. {key}"},
    "nl": {"title": "Jouw dependence(), voor elke waarde van marital_status",
           "x": "wat jouw dependence() geeft",
           "key": "Open stippen: wat jouw dependence geeft voor WOMEN, gevulde: voor MEN, bij elke waarde van "
                  "marital_status. Met Married-civ-spouse geeft hij {w} voor WOMEN en {m} voor MEN.",
           "edge": " Een waarde onder 0 of boven 1 staat aan de rand.",
           "means": " Dat is de gemiddelde kans op ja als elke vrouw, of elke man, die waarde had; zoals ze zijn {w0} en "
                    "{m0}.",
           "alt": "Twee stippen per waarde van marital_status, uit jouw dependence: vrouwen {women}; mannen {men}. "
                  "{key}"},
}


def dependences(fn, scores_of, women, men) -> tuple | None:
    """What `fn`, a dependence function, gives for women and for men at every value of marital_status."""
    w, m = [], []
    for v in MARITAL:
        a = _number(fn(scores_of, women.copy(), "marital_status", v))
        b = _number(fn(scores_of, men.copy(), "marital_status", v))
        if a is None or b is None:
            return None
        w.append(a)
        m.append(b)
    return w, m


def _dependence_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    SCORES.parts()
    got = _run(lambda: dependences(fn, SCORES, WOMEN.df, MEN.df))
    if got is None:
        return ""
    lang = seen.lang
    w = dict(_DEP_WORDS[lang])
    women, men = got
    i = MARITAL.index("Married-civ-spouse")
    key = w["key"].format(w=shown(women[i], lang), m=shown(men[i], lang))
    if any(v < 0 or v > 1 for v in women + men):
        key += w["edge"]
    if _same_as(list(got), list(_course("dependences", lambda: dependences(dependence, SCORES, WOMEN.df, MEN.df)))):
        key += w["means"].format(w0=number(SCORES(WOMEN.df).mean(), lang), m0=number(SCORES(MEN.df).mean(), lang))
    w["alt"] = w["alt"].format(women=", ".join(f"{v} {shown(x, lang)}" for v, x in zip(MARITAL, women)),
                               men=", ".join(f"{v} {shown(x, lang)}" for v, x in zip(MARITAL, men)), key=key)
    svg = two_groups(MARITAL, women, men, w, lang)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- the picture of exercise 3: your fewest_hours for six people with a no -------------------------------------------

# three women and three men of the test set with a no at their own hours
SIX_PEOPLE = [PERSON, 38495, 34820, 40293, 38092, 34175]

_HOURS_WORDS = {
    "en": {"title": "Your fewest_hours(), for six people of the test set with a no",
           "x": "hours a week", "woman": "woman", "man": "man",
           "key": "A row per person: the dot is the hours a week they work now, the arrow goes to what your "
                  "fewest_hours gives with top 99. {said}",
           "alt": "Six people with a no, with their hours now and what your fewest_hours gives. {key}"},
    "nl": {"title": "Jouw fewest_hours(), voor zes mensen uit de test set met een nee",
           "x": "uren per week", "woman": "vrouw", "man": "man",
           "key": "Een rij per persoon: de stip is hoeveel uur per week die persoon nu werkt, de pijl gaat naar wat "
                  "jouw fewest_hours geeft met top 99. {said}",
           "alt": "Zes mensen met een nee, met hun uren nu en wat jouw fewest_hours geeft. {key}"},
}


def _hours_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    SCORES.parts()
    people = table().loc[SIX_PEOPLE]

    def run():
        out = []
        for i in SIX_PEOPLE:
            got = fn(SCORES, people.loc[[i]].copy(), TOP)
            out.append(None if got is None else _number(got))
            if got is not None and out[-1] is None:
                return None
        return out

    got = _run(run)
    if got is None:
        return ""
    lang = seen.lang
    w = _HOURS_WORDS[lang]
    n = len(SIX_PEOPLE)
    p = chart.Panel(x=(0, 100), y=(-0.2, n + 0.2), xlabel=w["x"], ylabel="", yticks=[], xticks=[0, 20, 40, 60, 80, 100])
    said = []
    for k, i in enumerate(SIX_PEOPLE):
        row = people.loc[i]
        y = n - 1 - k + 0.3
        who = f"{w['woman'] if row['sex'] == 'Female' else w['man']}, {int(row['age'])}"
        now, then = int(row["hours_per_week"]), got[k]
        p.points([now], [y], role="data", r=4.5)
        if then is None:
            words = f"{who}: {now} → None"
        else:
            shown_at = min(max(then, 0), 100)
            if abs(shown_at - now) > 0.5:
                p.arrow(now, y, shown_at, y, role="pred", width=2)
            p.points([shown_at], [y], role="pred", r=4.5)
            words = f"{who}: {now} → {number(then, lang, 0) if float(then).is_integer() else number(then, lang, 2)}"
        p.text(0, y + 0.42, words, role="ink", anchor="start", halo=True)
        said.append(words)
    key = w["key"].format(said="; ".join(said) + ".")
    svg = chart.figure([p], alt=w["alt"].format(key=key), width=460, height=40 + 38 * n)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- the picture of The idea: where the gap between women and men sits in the course's classifier --------------------

_GAP_WORDS = {
    "en": {"title": "Where the gap in the logit sits: men minus women",
           "x": "mean Shapley value of the men minus that of the women (logit)",
           "key": "A bar per column: on the test set, the men's mean Shapley value on the logit minus the women's, "
                  "against the people to train on. The bars add up to {g}, the gap between the mean logit of the women "
                  "({lw}) and of the men ({lm}).",
           "alt": "A bar per column of the course's classifier: {all}. {key}"},
    "nl": {"title": "Waar het verschil in de logit zit: mannen min vrouwen",
           "x": "gemiddelde Shapley-waarde van de mannen min die van de vrouwen (logit)",
           "key": "Een staaf per kolom: op de test set de gemiddelde Shapley-waarde op de logit van de mannen min die "
                  "van de vrouwen, tegen de mensen om op te trainen. Samen zijn de staven {g}, het verschil tussen de "
                  "gemiddelde logit van de vrouwen ({lw}) en van de mannen ({lm}).",
           "alt": "Een staaf per kolom van de classifier van de course: {all}. {key}"},
}


def course_gap() -> tuple[dict, float, float]:
    """For the course's classifier on the test set: per column the men's mean Shapley value on the logit minus the
    women's, against the people to train on, and the mean logit of the women and of the men."""
    if "gap" not in _DATA:
        test = test_part()
        shares = shapley_logit(SCORES.parts(), test, train_part())
        mean = shares.groupby(test["sex"]).mean()
        gap = {c: float(mean.loc["Male", c] - mean.loc["Female", c]) for c in mean.columns}
        score = SCORES(test)
        z = pd.Series(np.log(score / (1 - score)), index=test.index).groupby(test["sex"]).mean()
        _DATA["gap"] = (gap, float(z["Female"]), float(z["Male"]))
    return _DATA["gap"]


def course_gap_figure(lang: str) -> str:
    gap, lw, lm = course_gap()
    w = dict(_GAP_WORDS[lang])
    key = w["key"].format(g=number(lm - lw, lang, 3), lw=number(lw, lang, 3), lm=number(lm, lang, 3))
    w["alt"] = w["alt"].format(all=", ".join(f"{c} {number(v, lang)}" for c, v in gap.items()), key=key)
    svg = bars(gap, w, lang, role="pred")
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "race" is a choice question, "report" an open question.
REQUIRED = ["importance", "dependence", "hours", "race", "report"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


# --- exercise 1: importance

_I_HEAD = ("def importance(scores_of, people, column, seed):\n"
           "    # how much less often the classifier is right on people when the column column is shuffled\n"
           "    shuffled = people.copy()\n"
           "    shuffled[column] = people[column].sample(frac=1, random_state=seed).to_numpy()\n")
_I_START = _I_HEAD + "    ...\n"
_I_RIGHT = '    right = ((scores_of(people) > 0.5) == (people["rich"] == 1)).mean()\n'
_I_AFTER = '    after = ((scores_of(shuffled) > 0.5) == (shuffled["rich"] == 1)).mean()\n'


def _i(right: str = _I_RIGHT, after: str = _I_AFTER, ret: str = "    return right - after\n") -> str:
    return _I_HEAD + right + after + ret


_I = _i()
_I_SAME = _i(after=_I_AFTER.replace("scores_of(shuffled)", "scores_of(people)").replace('shuffled["rich"]',
                                                                                        'people["rich"]'))
_I_SIGN = _i(ret="    return after - right\n")
_I_COUNT = _i(right=_I_RIGHT.replace(".mean()", ".sum()"), after=_I_AFTER.replace(".mean()", ".sum()"))
_I_YES = _i(right='    right = (scores_of(people) > 0.5).mean()\n', after='    after = (scores_of(shuffled) > 0.5).mean()\n')
_I_SHARE = _i(ret="    return (right - after) / right\n")

# --- exercise 2: dependence

_D_HEAD = ("def dependence(scores_of, people, column, value):\n"
           "    # the mean probability of a yes when everyone in people has value in the column column\n"
           "    changed = people.copy()\n")
_D_START = _D_HEAD + "    ...\n"


def _d(put: str = "    changed[column] = value\n", ret: str = "    return scores_of(changed).mean()\n") -> str:
    return _D_HEAD + put + ret


_D = _d()
_D_NOT_SET = _d(put="")
_D_SUM = _d(ret="    return scores_of(changed).sum()\n")
_D_YES = _d(ret="    return (scores_of(changed) > 0.5).mean()\n")
_D_FIRST = _d(ret="    return scores_of(changed)[0]\n")

# --- exercise 3: fewest_hours

_H_HEAD = ("def fewest_hours(scores_of, person, top):\n"
           "    # person: a table with one row; the fewest hours a week, from 1 to top, that give a yes, or None\n"
           "    changed = person.copy()\n"
           "    for hours in range(1, top + 1):\n")
_H_START = _H_HEAD + "        ...\n    return None\n"
_H_SET = '        changed["hours_per_week"] = hours\n'
_H_IF = "        if scores_of(changed)[0] > 0.5:\n"


def _h(put: str = _H_SET, cond: str = _H_IF, body: str = "            return hours\n") -> str:
    return _H_HEAD + put + cond + body + "    return None\n"


_H = _h()
_H_GE = _h(cond=_H_IF.replace("> 0.5", ">= 0.5"))
_H_NO = _h(cond=_H_IF.replace("> 0.5", "<= 0.5"))
_H_NOT_SET = _h(put="")

_LONG = 20.0   # seconds per case: the cases on PEOPLE score 16,281 people twice


EXERCISES = {
    "importance": FunctionBox(
        name="importance",
        picture=_importance_picture,
        function="importance",
        start=_I_START,
        solution=_I,
        save_as=("audit", "importance"),
        visible=[(SCORES, PEOPLE, "marital_status", 0)],
        hidden=[(SCORES, PEOPLE, "race", 1), (SCORES, MEN, "hours_per_week", 2), (BY_A, SIX, "a", 0),
                (BY_A, SIX, "b", 0), (BY_B, SIX, "a", 3)],
        digits=4,
        time_limit=_LONG,
        first=[_DOTS],
        mistakes=[
            Mistake("same", wrong=_I_SAME,
                    en="`{call}` gives {got}, and it should give {want}. Your function compares the people with themselves: `after` has to score `shuffled`, the table with the column shuffled, and compare with its `rich`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie vergelijkt de mensen met zichzelf: `after` moet `shuffled` scoren, de tabel met de geschudde kolom, en vergelijken met zijn `rich`."),
            Mistake("sign", wrong=_I_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: the right size, the wrong way round. The importance is how much *less* often the classifier is right after the shuffle: `right - after`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede grootte, andersom. De importance is hoeveel *minder* vaak de classifier goed zit na het schudden: `right - after`."),
            Mistake("count", wrong=_I_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: how many more people the classifier gets wrong, not the share. Take `.mean()` of the True and False, as in step one, so that tables of every size compare.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hoeveel mensen meer de classifier fout heeft, niet het deel. Neem `.mean()` van de True en False, zoals in stap één, zodat tabellen van elke grootte te vergelijken zijn."),
            Mistake("yes-rate", wrong=_I_YES,
                    en="`{call}` gives {got}, and it should give {want}: how much the share with a yes changes, not how often the classifier is right. Compare each yes, `scores_of(...) > 0.5`, with `rich == 1`, and take the mean of that.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hoeveel het deel met een ja verandert, niet hoe vaak de classifier goed zit. Vergelijk elke ja, `scores_of(...) > 0.5`, met `rich == 1`, en neem daar het gemiddelde van."),
            Mistake("share", wrong=_I_SHARE,
                    en="`{call}` gives {got}, and it should give {want}: the drop as a share of how often it was right. The importance here is the drop itself: `right - after`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de daling als deel van hoe vaak hij goed zat. De importance is hier de daling zelf: `right - after`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with three lines: `right`, how often the classifier is right on `people`; `after`, the same on `shuffled`; and `return right - after`.",
                      "nl": "Vervang de `...` door drie regels: `right`, hoe vaak de classifier goed zit op `people`; `after`, hetzelfde op `shuffled`; en `return right - after`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. On the people as they are, the classifier is right for 83.4% of them; shuffle `marital_status` with seed 0, and for 76.5%.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bij de mensen zoals ze zijn zit de classifier bij 83,4% goed; schud `marital_status` met seed 0, en bij 76,5%."},
            "almost": {"en": "Right for `importance(scores_of, PEOPLE, \"marital_status\", 0)`. But `{call}` gives {got}, and it should give {want}. Does your function work for every column, every seed and every table, and use only its arguments?",
                       "nl": "Goed voor `importance(scores_of, PEOPLE, \"marital_status\", 0)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke kolom, elke seed en elke tabel, en gebruikt hij alleen zijn argumenten?"},
            "correct": {"en": "Your `importance` is right in all {n} cases. Shuffle `marital_status`, and the classifier is right 0.070 less often; shuffle `race`, and it hardly changes. It is now in your project, in `{saved}`.",
                        "nl": "Je `importance` klopt in alle {n} gevallen. Schud `marital_status`, en de classifier zit 0,070 minder vaak goed; schud `race`, en er verandert bijna niets. Hij staat nu in je project, in `{saved}`."},
            "kept": _kept("importance"),
        },
    ),
    "dependence": FunctionBox(
        name="dependence",
        picture=_dependence_picture,
        function="dependence",
        start=_D_START,
        solution=_D,
        save_as=("audit", "dependence"),
        visible=[(SCORES, WOMEN, "marital_status", "Married-civ-spouse")],
        hidden=[(SCORES, MEN, "marital_status", "Married-civ-spouse"), (SCORES, WOMEN, "marital_status",
                                                                         "Never-married"),
                (SCORES, PEOPLE, "hours_per_week", 60), (BY_A, SIX, "a", 5), (BY_A, SIX, "b", 5)],
        digits=4,
        time_limit=_LONG,
        first=[_DOTS],
        mistakes=[
            Mistake("not-set", wrong=_D_NOT_SET,
                    en="`{call}` gives {got}, and it should give {want}: the mean probability of the people as they are. The column has to get `value` in the table you score: `changed[column] = value`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de gemiddelde kans van de mensen zoals ze zijn. De kolom moet `value` krijgen in de tabel die je scoort: `changed[column] = value`."),
            Mistake("sum", wrong=_D_SUM,
                    en="`{call}` gives {got}, and it should give {want}: the probabilities added up. A partial dependence is their mean: `.mean()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de kansen opgeteld. Een partial dependence is hun gemiddelde: `.mean()`."),
            Mistake("yes-rate", wrong=_D_YES,
                    en="`{call}` gives {got}, and it should give {want}: the share with a yes. The partial dependence is the mean probability itself, without the threshold of 0.5.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel met een ja. De partial dependence is de gemiddelde kans zelf, zonder de threshold van 0,5."),
            Mistake("first", wrong=_D_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: the probability of the first person only. Take the mean over everyone: `.mean()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de kans van de eerste persoon. Neem het gemiddelde over iedereen: `.mean()`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: give the column `column` the value `value` in `changed`, and return the mean of `scores_of(changed)`.",
                      "nl": "Vervang de `...` door twee regels: geef de kolom `column` de waarde `value` in `changed`, en geef het gemiddelde van `scores_of(changed)` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. As they are, the women of the test set have a mean probability of a yes of 0.126; with everyone `Married-civ-spouse`, of 0.318.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Zoals ze zijn, hebben de vrouwen van de test set een gemiddelde kans op ja van 0,126; met iedereen `Married-civ-spouse` van 0,318."},
            "almost": {"en": "Right for the women. But `{call}` gives {got}, and it should give {want}. Does your function work for every table, every column and every value?",
                       "nl": "Goed voor de vrouwen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tabel, elke kolom en elke waarde?"},
            "correct": {"en": "Your `dependence` is right in all {n} cases. If every woman of the test set were Married-civ-spouse, their mean probability of a yes would be 0.318 instead of 0.126; for the men, 0.381 instead of 0.295. It is now in your project, in `{saved}`.",
                        "nl": "Je `dependence` klopt in alle {n} gevallen. Als elke vrouw van de test set Married-civ-spouse was, zou hun gemiddelde kans op ja 0,318 zijn in plaats van 0,126; voor de mannen 0,381 in plaats van 0,295. Hij staat nu in je project, in `{saved}`."},
            "kept": _kept("dependence"),
        },
    ),
    "hours": FunctionBox(
        name="hours",
        picture=_hours_picture,
        function="fewest_hours",
        start=_H_START,
        solution=_H,
        save_as=("audit", "recourse"),
        visible=[(SCORES, ONE, 99)],
        hidden=[(SCORES, ONE, 60), (SCORES, MARRIED, 99), (BY_HOURS, ROW, 99), (FEW, ROW, 40), (BY_HOURS, ROW, 30)],
        time_limit=_LONG,
        first=[_DOTS],
        mistakes=[
            Mistake("ge", wrong=_H_GE,
                    en="`{call}` gives {got}, and it should give {want}. At exactly 0.5 the classifier says no: a yes is a probability above 0.5, so use `> 0.5`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij precies 0,5 zegt de classifier nee: een ja is een kans boven 0,5, dus gebruik `> 0.5`."),
            Mistake("no", wrong=_H_NO,
                    en="`{call}` gives {got}, and it should give {want}: the fewest hours that give a *no*. Ask whether the probability is above 0.5: `> 0.5`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het kleinste aantal uren dat een *nee* geeft. Vraag of de kans boven 0,5 ligt: `> 0.5`."),
        ],
        texts=_SAME_TEXTS | {
            "unreached": {"en": "`{call}` gives `None`, and it should give a number: some number of hours up to `top` does give this person a yes. Does `changed` get the hours of each round, `changed[\"hours_per_week\"] = hours`, before you score it? And is `return None` after the loop, not in it?",
                          "nl": "`{call}` geeft `None`, en dat moet een getal zijn: een aantal uren tot `top` geeft deze persoon wel een ja. Krijgt `changed` de uren van elke ronde, `changed[\"hours_per_week\"] = hours`, voordat je hem scoort? En staat `return None` na de loop, niet erin?"},
            "empty": {"en": "Replace the `...` with three lines: give `changed` the hours of this round, ask whether the probability of `changed` is above 0.5, and if so return `hours`.",
                      "nl": "Vervang de `...` door drie regels: geef `changed` de uren van deze ronde, vraag of de kans van `changed` boven 0,5 ligt, en geef dan `hours` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The woman of the worked example works 50 hours a week and has a probability of 0.278; from 82 hours a week on, the probability is above 0.5.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De vrouw uit het uitgewerkte voorbeeld werkt 50 uur per week en heeft een kans van 0,278; vanaf 82 uur per week ligt de kans boven 0,5."},
            "almost": {"en": "Right for `fewest_hours(scores_of, ONE, 99)`. But `{call}` gives {got}, and it should give {want}. Does your function give `None` when no number of hours up to `top` gives a yes, and work for a person who would get a yes with fewer hours than now?",
                       "nl": "Goed voor `fewest_hours(scores_of, ONE, 99)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie `None` als geen aantal uren tot `top` een ja geeft, en werkt hij voor iemand die met minder uren dan nu een ja zou krijgen?"},
            "correct": {"en": "Your `fewest_hours` is right in all {n} cases. The woman of the worked example would get a yes from 82 hours a week on, 32 more than now; with at most 60 hours, never. It is now in your project, in `{saved}`.",
                        "nl": "Je `fewest_hours` klopt in alle {n} gevallen. De vrouw uit het uitgewerkte voorbeeld zou vanaf 82 uur per week een ja krijgen, 32 meer dan nu; met hooguit 60 uur nooit. Hij staat nu in je project, in `{saved}`."},
            "kept": _kept("fewest_hours"),
        },
    ),
}


# --- what your project can do now: the explanations of your own classifier ----------------------------------------

_DEMO_RUNS: dict = {}
_MODULES = ("model", "report", "importance", "dependence", "recourse")


def _project_texts() -> tuple:
    texts = []
    for module in _MODULES:
        path = projects.folder() / "audit" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo() -> dict | str | None:
    """The explanations of your own classifier (audit/model.py): your report of step one per sex, your importance per
    column over three seeds, the Shapley values of the logit per sex (by the course, from your classifier's weights),
    your dependence per value of marital_status for women and men, and your fewest_hours for 100 women and 100 men
    with a no. None before exercise 1; the name of the module that stops with an error or gives something else."""
    texts = _project_texts()
    if texts[2] is None:
        return None
    if texts in _DEMO_RUNS:
        return _DEMO_RUNS[texts]
    projects.forget()
    out = _demo()
    _DEMO_RUNS.clear()
    _DEMO_RUNS[texts] = out
    return out


def _learner_scorer(people: pd.DataFrame):
    """scores_of for the learner's own model.py, and whether it is the fast one. The fast one makes the columns as
    the course's features() does; when the learner changed features() itself, it would differ, and the slow one asks
    features() for the whole table with the rows added."""
    from everything_ai.check import run_code
    run = run_code("import audit.model as model_module")
    if run.error is not None:
        return None, False
    module = run.names["model_module"]
    try:
        parts = checking.with_time_limit(lambda: _parts_of(module, people), 120)
        fast = Scorer(lambda: parts)
        mine = checking.with_time_limit(lambda: module.scores(people.copy()), 120)
        if np.allclose(fast(people), np.asarray(mine, dtype=float), atol=1e-9):
            return fast, True
    except Exception:   # also Timeout, and a model.py without NUMBERS or WORDS
        pass
    try:
        model = checking.with_time_limit(lambda: module.train(people), 120)
    except Exception:
        return None, False

    def slow(t):
        X = module.features(pd.concat([people, t]))
        return model.predict_proba(X.iloc[len(people):])[:, 1]
    return Fake(slow, "scores_of"), False


def _demo() -> dict | str:
    from everything_ai.check import run_code
    have = {m: (projects.folder() / "audit" / f"{m}.py").exists() for m in _MODULES}
    if not have["model"]:
        return "model"
    people = table().copy()
    scores_of, fast = _learner_scorer(people)
    if scores_of is None:
        return "model"
    try:
        people["score"] = [float(s) for s in checking.with_time_limit(lambda: scores_of(people), 120)]
    except Exception:
        return "model"
    if not all(0 <= s <= 1 for s in people["score"]):
        return "model"
    people["yes"] = people["score"] > 0.5
    test = people[people["split"] == "test"]
    out = {"right": float((test["yes"] == (test["rich"] == 1)).mean()), "fast": fast}

    run = run_code("from audit.report import report")
    try:
        if run.error is not None:
            raise ValueError("no report")
        out["report"] = checking.with_time_limit(lambda: run.names["report"](test.copy(), "sex"), 30)
        if not isinstance(out["report"], pd.DataFrame) or "yes rate" not in out["report"].columns:
            raise ValueError("no report")
    except Exception:
        out["report"] = "report"

    run = run_code("from audit.importance import importance")
    if run.error is not None:
        return "importance"
    seeds = list(range(SEED_COUNT if fast else 1))
    got = _run(lambda: importances(run.names["importance"], scores_of, test, seeds), 300)
    if got is None:
        return "importance"
    out["importance"] = got

    parts = scores_of.parts() if isinstance(scores_of, Scorer) else None
    shares = None if parts is None else shapley_logit(parts, test, people[people["split"] == "train"])
    if shares is not None:
        mean = shares.groupby(test["sex"]).mean()
        if {"Female", "Male"} <= set(mean.index):
            out["gap"] = {c: float(mean.loc["Male", c] - mean.loc["Female", c]) for c in mean.columns}
            z = np.log(test["score"] / (1 - test["score"]))
            means = z.groupby(test["sex"]).mean()
            out["logits"] = (float(means["Female"]), float(means["Male"]))

    women, men = test[test["sex"] == "Female"], test[test["sex"] == "Male"]
    out["dependence"] = None
    if have["dependence"]:
        run = run_code("from audit.dependence import dependence")
        got = None if run.error is not None else _run(
            lambda: dependences(run.names["dependence"], scores_of, women, men), 300)
        out["dependence"] = got if got is not None else "dependence"
        out["means"] = (float(women["score"].mean()), float(men["score"].mean()))

    out["recourse"] = None
    if have["recourse"]:
        run = run_code("from audit.recourse import fewest_hours")
        size = SAMPLE if fast else 10
        try:
            if run.error is not None:
                raise ValueError("no recourse")
            found = {}
            for sex, group in (("Female", women), ("Male", men)):
                no = group[~group["yes"]]
                some = no.sample(min(size, len(no)), random_state=0)
                needs = []
                for i in some.index:
                    got = checking.with_time_limit(lambda: run.names["fewest_hours"](scores_of, some.loc[[i]].copy(),
                                                                                       TOP), 30)
                    if got is not None and _number(got) is None:
                        raise ValueError("not a number")
                    needs.append(None if got is None else _number(got))
                hours = [n for n in needs if n is not None]
                extra = [n - h for n, h in zip(needs, some["hours_per_week"]) if n is not None]
                found[sex] = {"people": len(some), "with": len(hours),
                              "median": float(np.median(hours)) if hours else None,
                              "extra": float(np.median(extra)) if extra else None}
            out["recourse"] = found
        except Exception:
            out["recourse"] = "recourse"
    return out


_DEMO_WORDS = {
    "en": {
        "names": {"Female": "women", "Male": "men"},
        "right": "Your classifier is right for {r}% of the people of the test set.",
        "report": "**Step one.** Your `report` per sex on the test set: a yes rate of {w} for the women and {m} for "
                  "the men, an impact ratio of {i} for the women.",
        "report-missing": "**Step one.** Your `report` is not there, or does not give a column `yes rate`, so this "
                          "part is left out. It comes from exercise 1 of *Project: measure the bias of your own "
                          "classifier*.",
        "imp-title": "Your importance per column, mean of {k} {seeds}",
        "seed": "seed", "seeds": "seeds",
        "imp-x": "what your importance() gives on the test set",
        "imp-alt": "A bar per column of your classifier, from your importance on the test set: {all}.",
        "imp-text": "Your importance gives the most to **{best}**: {bv}. **race** gets {race}.",
        "gap-title": "Where the gap in the logit sits: men minus women",
        "gap-x": "mean Shapley value of the men minus that of the women (logit)",
        "gap-alt": "A bar per column: how much higher the men's mean Shapley value of the logit is than the "
                   "women's. {all}",
        "gap-text": "The course worked these out from your classifier's weights, against the people to train on: the "
                    "mean logit of the women is {lw}, of the men {lm}, a gap of {g}. **{best}** carries {bv} of it.",
        "gap-none": "*Your classifier is not a logistic regression with one weight per column, so the course does not "
                    "work out its Shapley values here.*",
        "dep-title": "Your dependence per value of marital_status",
        "dep-x": "what your dependence() gives",
        "dep-alt": "Two dots per value of marital_status, from your dependence: women {women}; men {men}.",
        "dep-text": "Open dots: the women of the test set, filled: the men. As they are, their mean probability of a "
                    "yes is {w} and {m}; your dependence gives {dw} and {dm} with everyone Married-civ-spouse.",
        "rec-text": "**Recourse.** Of {nw} women with a no, drawn at random, your `fewest_hours` gives a number of "
                    "hours for {hw}{mw}, and `None` for the others; of {nm} men with a no, for {hm}{mm}.",
        "rec-median": " (middle value {h} hours; middle value of the extra hours {e})",
        "slow": "*Your `features()` in `model.py` makes other columns than the course's, so the demo asks it for "
                "every table and runs fewer cases: one seed and 10 people per sex.*",
        "todo": "*Exercise {n} is not done yet: its part follows once its check is right.*",
    },
    "nl": {
        "names": {"Female": "vrouwen", "Male": "mannen"},
        "right": "Je classifier zit goed bij {r}% van de mensen in de test set.",
        "report": "**Stap één.** Jouw `report` per geslacht op de test set: een deel met een ja van {w} bij de vrouwen "
                  "en {m} bij de mannen, een impact ratio van {i} voor de vrouwen.",
        "report-missing": "**Stap één.** Jouw `report` ontbreekt, of geeft geen kolom `yes rate`, dus dit deel valt "
                          "weg. Hij komt uit oefening 1 van *Project: meet de bias van je eigen classifier*.",
        "imp-title": "Jouw importance per kolom, gemiddelde van {k} {seeds}",
        "seed": "seed", "seeds": "seeds",
        "imp-x": "wat jouw importance() geeft op de test set",
        "imp-alt": "Een staaf per kolom van je classifier, uit jouw importance op de test set: {all}.",
        "imp-text": "Jouw importance geeft het meest aan **{best}**: {bv}. **race** krijgt {race}.",
        "gap-title": "Waar het verschil in de logit zit: mannen min vrouwen",
        "gap-x": "gemiddelde Shapley-waarde van de mannen min die van de vrouwen (logit)",
        "gap-alt": "Een staaf per kolom: hoeveel hoger de gemiddelde Shapley-waarde van de logit van de mannen is dan "
                   "die van de vrouwen. {all}",
        "gap-text": "De course rekende deze uit met de weights van je classifier, tegen de mensen om op te trainen: de "
                    "gemiddelde logit van de vrouwen is {lw}, van de mannen {lm}, een verschil van {g}. **{best}** "
                    "draagt daarvan {bv}.",
        "gap-none": "*Je classifier is geen logistische regressie met één weight per kolom, dus de course rekent hier "
                    "geen Shapley-waarden uit.*",
        "dep-title": "Jouw dependence per waarde van marital_status",
        "dep-x": "wat jouw dependence() geeft",
        "dep-alt": "Twee stippen per waarde van marital_status, uit jouw dependence: vrouwen {women}; mannen {men}.",
        "dep-text": "Open stippen: de vrouwen van de test set, gevulde: de mannen. Zoals ze zijn, is hun gemiddelde kans "
                    "op ja {w} en {m}; jouw dependence geeft {dw} en {dm} met iedereen Married-civ-spouse.",
        "rec-text": "**Recourse.** Van {nw} willekeurig gekozen vrouwen met een nee geeft jouw `fewest_hours` bij {hw} "
                    "een aantal uren{mw}, en `None` bij de anderen; van {nm} mannen met een nee bij {hm}{mm}.",
        "rec-median": " (middelste waarde {h} uur; middelste waarde van de extra uren {e})",
        "slow": "*Jouw `features()` in `model.py` maakt andere kolommen dan die van de course, dus de demo vraagt hem "
                "om elke tabel en doet minder gevallen: één seed en 10 mensen per geslacht.*",
        "todo": "*Oefening {n} is nog niet af: zijn deel volgt zodra de check goed is.*",
    },
}

_BROKEN = {
    "en": {"model": "*Your classifier in `work/projects/audit/model.py` stops with an error, or does not give one probability from 0 to 1 per person. Did you change the file? Undo your change, or delete the file and open the lesson again: the course then puts its own version back.*",
           "importance": "*Your `importance` in `work/projects/audit/importance.py` stops with an error, or does not give a number. Did you change its file? Delete it, and check exercise 1 again: its check then saves your code again.*",
           "dependence": "*Your `dependence` in `work/projects/audit/dependence.py` stops with an error, or does not give a number. Did you change its file? Delete it, and check exercise 2 again.*",
           "recourse": "*Your `fewest_hours` in `work/projects/audit/recourse.py` stops with an error, or does not give a number or `None`. Did you change its file? Delete it, and check exercise 3 again.*"},
    "nl": {"model": "*Je classifier in `work/projects/audit/model.py` stopt met een fout, of geeft niet per persoon één kans van 0 tot 1. Heb je het bestand veranderd? Maak je wijziging ongedaan, of verwijder het bestand en open de les opnieuw: de course zet dan zijn eigen versie terug.*",
           "importance": "*Je `importance` in `work/projects/audit/importance.py` stopt met een fout, of geeft geen getal. Heb je het bestand veranderd? Verwijder het, en check oefening 1 opnieuw: de check bewaart je code dan weer.*",
           "dependence": "*Je `dependence` in `work/projects/audit/dependence.py` stopt met een fout, of geeft geen getal. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw.*",
           "recourse": "*Je `fewest_hours` in `work/projects/audit/recourse.py` stopt met een fout, of geeft geen getal of `None`. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*"},
}


def broken(name: str, lang: str) -> str:
    return _BROKEN[lang][name]


def _share(x: float, lang: str) -> str:
    return number(x, lang, 3)


def demo_parts(audit: dict, lang: str) -> list:
    """What the lesson shows of the demo: a list of ("md", text) and ("html", figure)."""
    w = _DEMO_WORDS[lang]
    parts = [("md", w["right"].format(r=number(100 * audit["right"], lang, 1)))]
    if not audit["fast"]:
        parts.append(("md", w["slow"]))
    rep = audit["report"]
    rates = None
    if isinstance(rep, pd.DataFrame) and {"Female", "Male"} <= set(rep.index):
        a, b = _number(rep.loc["Female", "yes rate"]), _number(rep.loc["Male", "yes rate"])
        if a is not None and b is not None and max(a, b) > 0:
            rates = (a, b)
    if rates is None:
        parts.append(("md", w["report-missing"]))
    else:
        parts.append(("md", w["report"].format(w=number(rates[0], lang, 3), m=number(rates[1], lang, 3),
                                               i=number(rates[0] / max(rates), lang, 2))))

    imp = audit["importance"]
    k = SEED_COUNT if audit["fast"] else 1
    best = max(imp, key=imp.get)
    words = {"x": w["imp-x"], "alt": w["imp-alt"].format(all=", ".join(f"{c} {number(v, lang)}" for c, v in
                                                                        imp.items()))}
    title = w["imp-title"].format(k=k, seeds=w["seed"] if k == 1 else w["seeds"])
    text = w["imp-text"].format(best=best, bv=number(imp[best], lang), race=number(imp.get("race", 0.0), lang))
    parts.append(("html", yours.figure(title, bars(imp, words, lang, role="loss")
                                       + f'<div style="{_UI}">{_md_inline(text)}</div>')))

    if "gap" in audit:
        gap = audit["gap"]
        best = max(gap, key=gap.get)
        lw, lm = audit["logits"]
        words = {"x": w["gap-x"], "alt": w["gap-alt"].format(all=", ".join(f"{c} {number(v, lang)}" for c, v in
                                                                            gap.items()))}
        text = w["gap-text"].format(lw=number(lw, lang, 2), lm=number(lm, lang, 2), g=number(lm - lw, lang, 2),
                                    best=best, bv=number(gap[best], lang, 2))
        parts.append(("html", yours.figure(w["gap-title"], bars(gap, words, lang, role="pred")
                                           + f'<div style="{_UI}">{_md_inline(text)}</div>')))
    else:
        parts.append(("md", w["gap-none"]))

    dep = audit["dependence"]
    if dep is None:
        parts.append(("md", w["todo"].format(n=2)))
    elif isinstance(dep, str):
        parts.append(("md", broken(dep, lang)))
    else:
        women, men = dep
        i = MARITAL.index("Married-civ-spouse")
        mw, mm = audit["means"]
        words = {"x": w["dep-x"], "alt": w["dep-alt"].format(
            women=", ".join(f"{v} {number(x, lang)}" for v, x in zip(MARITAL, women)),
            men=", ".join(f"{v} {number(x, lang)}" for v, x in zip(MARITAL, men)))}
        text = w["dep-text"].format(w=number(mw, lang), m=number(mm, lang), dw=number(women[i], lang),
                                    dm=number(men[i], lang))
        parts.append(("html", yours.figure(w["dep-title"], two_groups(MARITAL, women, men, words, lang)
                                           + f'<div style="{_UI}">{_md_inline(text)}</div>')))

    rec = audit["recourse"]
    if rec is None:
        parts.append(("md", w["todo"].format(n=3)))
    elif isinstance(rec, str):
        parts.append(("md", broken(rec, lang)))
    else:
        def middle(r):
            if r["median"] is None:
                return ""
            return w["rec-median"].format(h=number(r["median"], lang, 0 if float(r["median"]).is_integer() else 1),
                                          e=number(r["extra"], lang, 0 if float(r["extra"]).is_integer() else 1))
        f, m = rec["Female"], rec["Male"]
        parts.append(("md", w["rec-text"].format(nw=f["people"], hw=f["with"], mw=middle(f), nm=m["people"],
                                                 hm=m["with"], mm=middle(m))))
    return parts


def _md_inline(text: str) -> str:
    """**bold** and `code` of a short text as HTML, the rest escaped."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", out)


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("importance", _I_START, "empty", None),
    ("importance", _I, "correct", None),
    ("importance", _I_HEAD + '    rich = people["rich"] == 1\n    before = (scores_of(people) > 0.5) == rich\n'
                   '    later = (scores_of(shuffled) > 0.5) == rich\n    return before.mean() - later.mean()\n',
     "correct", None),
    ("importance", _I_SAME, "wrong", "same"),
    ("importance", _I_SIGN, "wrong", "sign"),
    ("importance", _I_COUNT, "wrong", "count"),
    ("importance", _I_YES, "wrong", "yes-rate"),
    ("importance", _I_SHARE, "wrong", "share"),
    ("importance", _I_HEAD + "    return 0.06965174129353235\n", "almost", None),
    ("importance", _i(ret="    return\n"), "wrong", "no-return"),
    ("importance", _I_HEAD + "    ...\n    return 0.06965174129353235\n", "wrong", "dots-left"),
    ("dependence", _D_START, "empty", None),
    ("dependence", _D, "correct", None),
    ("dependence", _D_HEAD + "    changed[column] = value\n    scores = scores_of(changed)\n"
                   "    return sum(scores) / len(scores)\n", "correct", None),
    ("dependence", _D_NOT_SET, "wrong", "not-set"),
    ("dependence", _D_SUM, "wrong", "sum"),
    ("dependence", _D_YES, "wrong", "yes-rate"),
    ("dependence", _D_FIRST, "wrong", "first"),
    ("dependence", _D_HEAD + "    return 0.31816430550500796\n", "almost", None),
    ("dependence", _d(ret="    return\n"), "wrong", "no-return"),
    ("dependence", _D_HEAD + "    ...\n    return 0.31816430550500796\n", "wrong", "dots-left"),
    ("hours", _H_START, "empty", None),
    ("hours", _H, "correct", None),
    ("hours", _H_HEAD + _H_SET + "        yes = scores_of(changed) > 0.5\n        if yes[0]:\n"
              "            return hours\n    return None\n", "correct", None),
    ("hours", _H_GE, "wrong", "ge"),
    ("hours", _H_NO, "wrong", "no"),
    ("hours", _H_NOT_SET, "wrong", "unreached"),
    ("hours", _h(body="            return hours\n        return None\n"), "wrong", "unreached"),
    ("hours", "def fewest_hours(scores_of, person, top):\n    return 82\n", "almost", None),
    ("hours", _H_HEAD + "        ...\n    return 82\n", "wrong", "dots-left"),
]
