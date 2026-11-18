# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-equalized-odds-calibration: the recall and the false positive rate per sex of a made-up
rule on the table UCI Adult, the false positive rates that equal precision and equal recall force on two groups with
different base rates, and sorting claims by what they compare between groups; with the pictures of how well the
model's probabilities mean the same for women and men, of its errors with one limit and with equal recall, and of the
learner's own rates."""
import math
import numbers
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import paths, yours

DATA = paths.ROOT / "data" / "adult.csv.gz"
NUMBERS = ["age", "education_num", "hours_per_week"]
WORDS = ["workclass", "marital_status", "occupation", "relationship", "race"]   # the word columns, without sex
COLUMN = "sex"
OTHER_COLUMN = "race"
SCHOOL = 13   # the made-up rule: a yes for an education_num of at least this (a bachelor's degree)

# What the demo works out on the test part of the table (tests/test_rai_equalized_odds_calibration.py checks it).
BASE = (0.1088, 0.2998)   # the share of women and of men who really earn more than 50,000 dollars
# Per way of choosing: the recall, the false positive rate and the precision, each for women and men.
ERRORS = {"one": ((0.4356, 0.5697), (0.0259, 0.1101), (0.6728, 0.6891)),
          "equal": ((0.5695, 0.5697), (0.0493, 0.1101), (0.5854, 0.6891))}
LIMIT_WOMEN = 0.3679   # the women's limit for equal recall; the men keep 0.5
# Per band of probabilities: the average probability in it, and the share that really earns more, for women and men.
BANDS = {"women": ((0.0467, 0.0325), (0.2829, 0.2486), (0.4949, 0.4508), (0.7005, 0.6907), (0.8576, 0.8795)),
         "men": ((0.0592, 0.0635), (0.2944, 0.3186), (0.4948, 0.5021), (0.7020, 0.6859), (0.8591, 0.8441))}
EDGES = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]

# Exercise 2: the base rates of the test part, and one precision and one recall for both groups.
P = {"women": 0.109, "men": 0.300}
PRECISION = 0.69
RECALL = 0.57

_TABLE = {}


def people():
    """The table UCI Adult of the course, read once."""
    if "people" not in _TABLE:
        import pandas as pd
        _TABLE["people"] = pd.read_csv(DATA)
    return _TABLE["people"]


def model_scores():
    """What the demo trains: the model without the column sex, and its probability of a yes for each person of the
    test part, as (the test part, the probabilities)."""
    if "scores" not in _TABLE:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        table = people()
        train, test = table["split"] == "train", table["split"] == "test"
        X = pd.get_dummies(table[NUMBERS + WORDS], columns=WORDS, dtype=float)
        X = X / X.max()
        with threadpool_limits(1, user_api="openmp"):
            model = LogisticRegression(max_iter=1000)
            model.fit(X[train], table["income"][train] == ">50K")
            _TABLE["scores"] = (table[test], model.predict_proba(X[test])[:, 1])
    return _TABLE["scores"]


def ways() -> dict:
    """The two ways of the demo: one limit of 0.5, and a lower limit for women so that their recall is the men's;
    with "limit" the women's limit."""
    import numpy as np
    tested, score = model_scores()
    rich = tested["income"] == ">50K"
    women, men = tested["sex"] == "Female", tested["sex"] == "Male"
    one = score > 0.5
    recall_men = one[men * rich].mean()
    limit_women = np.quantile(score[women * rich], 1 - recall_men)
    return {"one": one, "equal": score > limit_women * women + 0.5 * men, "limit": float(limit_women)}


def errors(yes) -> tuple:
    """For a yes per person of the test part: the recall, the false positive rate and the precision, each as
    (women, men)."""
    tested, _ = model_scores()
    rich, poor = tested["income"] == ">50K", tested["income"] == "<=50K"
    out = ([], [], [])
    for sex in ("Female", "Male"):
        group = tested["sex"] == sex
        out[0].append(float(yes[group * rich].mean()))
        out[1].append(float(yes[group * poor].mean()))
        out[2].append(float(rich[group * yes].mean()))
    return tuple(tuple(x) for x in out)


def bands() -> dict:
    """Per sex and per band of EDGES: the average probability in the band, and the share that really earns more."""
    tested, score = model_scores()
    rich = tested["income"] == ">50K"
    out = {}
    for key, sex in (("women", "Female"), ("men", "Male")):
        group = tested["sex"] == sex
        rows = []
        for low, high in EDGES:
            band = (score >= low) * (score < high) * group
            rows.append((float(score[band].mean()), float(rich[band].mean())))
        out[key] = tuple(rows)
    return out


def rule_rates(column: str, who: str = "rich", rule: str = "school", over: str = "group") -> dict:
    """Per value of `column`, for the rows of `who` ("rich", "poor" or "all"): the share that the rule gives a yes
    ("school"; "strict" with > instead of >=; "no" for the share that gets no yes). With over="yes", the share of
    the rows of `who` among all the yes of the group instead."""
    table = people()
    rows = {"rich": table[table["income"] == ">50K"], "poor": table[table["income"] == "<=50K"], "all": table}[who]
    if rule == "school":
        picked = rows[rows["education_num"] >= SCHOOL]
    elif rule == "strict":
        picked = rows[rows["education_num"] > SCHOOL]
    else:
        picked = rows[rows["education_num"] < SCHOOL]
    if over == "yes":
        yes = table[table["education_num"] >= SCHOOL]
        s = picked.groupby(column).size() / yes.groupby(column).size()
    else:
        s = picked.groupby(column).size() / rows.groupby(column).size()
    return {k: float(v) for k, v in s.items()}


def forced(p: dict, precision: float, recall: float) -> dict:
    """The false positive rate each group must have, per 1,000 people, with this precision and recall."""
    out = {}
    for group, share in p.items():
        positives = 1000 * share
        true_yes = recall * positives
        false_yes = true_yes / precision - true_yes
        out[group] = false_yes / (1000 - positives)
    return out


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _num(x: float, decimal: str) -> str:
    return f"{x:.2f}".replace(".", decimal)


def picture_meaning(labels: dict) -> str:
    """How well the model's probabilities mean the same for women (open dots) and men (filled dots): per band, across
    the average probability and up the share that really earns more, with a dashed line where the two are equal.
    `labels` gives "title", "xlabel", "ylabel", "line", "key", "decimal" and "alt"."""
    from everything_ai import chart
    ticks = [0, 0.25, 0.5, 0.75, 1]
    names = [f"{t:g}" for t in ticks]
    p = chart.Panel(x=(0, 1), y=(0, 1), xlabel=labels["xlabel"], ylabel=labels["ylabel"], xticks=ticks,
                    xtick_labels=names, yticks=ticks)
    p.line([0, 1], [0, 1], role="ink-3", width=1, dashed=True)
    p.text(0.62, 0.53, labels["line"], "ink-3", anchor="start", halo=True)
    for key, hollow in (("women", True), ("men", False)):
        xs, ys = zip(*BANDS[key])
        p.points(xs, ys, role="pred", r=5, hollow=hollow)
    svg = chart.figure([p], alt=labels["alt"], width=420, height=330)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def picture_errors(labels: dict) -> str:
    """Three panels with a row per way of choosing, in the order of ERRORS: the recall, the false positive rate and
    the precision of women (open bars) and men (filled bars). `labels` gives "title", "rows" (two names), "xlabels"
    (three), "key", "decimal" and "alt"."""
    from everything_ai import chart
    keys = list(ERRORS)
    top = 1.6 * len(keys) + 0.1
    panels = []
    for k, (role, xmax, ticks) in enumerate((("data", 1, [0, 0.5, 1]), ("loss", 0.2, [0, 0.1, 0.2]),
                                            ("data", 1, [0, 0.5, 1]))):
        p = chart.Panel(x=(0, xmax), y=(0, top), xlabel=labels["xlabels"][k], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
        for i, (key, name) in enumerate(zip(keys, labels["rows"])):
            women, men = ERRORS[key][k]
            y = top - 1.6 * (i + 1)
            if k == 0:
                p.text(0.002, y + 1.2, name, "ink", anchor="start")
            p.rect(0, y + 0.55, women, y + 0.85, role=role)
            p.rect(0, y + 0.15, men, y + 0.45, role=role, solid=True)
            gap = 0.015 * xmax
            p.text(women + gap, y + 0.7, _num(women, labels["decimal"]), "ink-2", anchor="start")
            p.text(men + gap, y + 0.3, _num(men, labels["decimal"]), "ink-2", anchor="start")
        panels.append(p)
    svg = chart.figure(panels, alt=labels["alt"], width=240, height=230)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    return float(x)


def _rates(v) -> dict | None:
    """The learner's Series of a share per value, when it is one: 1 to 12 values, each a number."""
    import pandas as pd
    if not isinstance(v, pd.Series) or not 1 <= len(v) <= 12:
        return None
    out = {}
    for k, x in v.items():
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        if not isinstance(k, str):
            return None
        out[k] = x
    return out


def _two(v) -> tuple | None:
    """The learner's two Series, when the last line gives two."""
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        return None
    a, b = _rates(v[0]), _rates(v[1])
    return None if a is None or b is None else (a, b)


def _same(a: dict, b: dict) -> bool:
    """Two shares per value, the same values, each the same to two decimals."""
    return (set(a) == set(b) and all(math.isfinite(a[k]) and abs(a[k] - b[k]) < 0.006 for k in a))


def _close(got, want) -> bool:
    """The recall and the false positive rate per group: rounded or not."""
    a, b = _two(got), _two(want)
    return a is not None and b is not None and _same(a[0], b[0]) and _same(a[1], b[1])


def _want(column: str = COLUMN) -> tuple:
    return rule_rates(column, "rich"), rule_rates(column, "poor")


def _fpr_is(make):
    """A mistake where the recall is right and the false positive rate is what `make()` gives for the column sex."""
    def match(got, text):
        two = _two(got)
        return two is not None and _same(two[0], _want()[0]) and _same(two[1], make())
    return match


def _both_are(make_tpr, make_fpr):
    def match(got, text):
        two = _two(got)
        return two is not None and _same(two[0], make_tpr()) and _same(two[1], make_fpr())
    return match


def _complement(got, text) -> bool:
    two = _two(got)
    if two is None or not _same(two[0], _want()[0]):
        return False
    return _same(two[1], {k: 1 - v for k, v in _want()[0].items()})


def _counts(got, text) -> bool:
    two = _two(got)
    return two is not None and any(x > 1 for part in two for x in part.values())


def _missing(got, text) -> bool:
    """Two Series, one of them empty or with a nan: a selection without rows."""
    import pandas as pd
    if not isinstance(got, (tuple, list)) or len(got) != 2:
        return False
    if not all(isinstance(v, pd.Series) for v in got):
        return False
    return any(len(v) == 0 or bool(v.isna().any()) for v in got)


def _series(got, text) -> bool:
    import pandas as pd
    return isinstance(got, pd.Series)


def _one_thing(got, text) -> bool:
    return _number(got) is not None


_WORDS = {
    "en": {"title": "Your error rates of the rule per value of {column}", "tpr": "recall", "fpr": "false positive rate",
           "alt": "Two panels with one bar per value of {column}. Left, the recall: {tpr}. Right, the false positive rate: {fpr}."},
    "nl": {"title": "Jouw foutpercentages van de regel per waarde van {column}", "tpr": "recall",
           "fpr": "false positive rate",
           "alt": "Twee panelen met één balk per waarde van {column}. Links de recall: {tpr}. Rechts de false positive rate: {fpr}."},
}


def _errors_picture(seen) -> str:
    """Per value of the column, a bar as long as the learner's recall (left) and false positive rate (right), with
    dashed lines at the lowest and the highest of each."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    tpr, fpr = _rates(names.get("tpr")), _rates(names.get("fpr"))
    column = names.get("column", COLUMN)
    if tpr is None or fpr is None or set(tpr) != set(fpr) or not isinstance(column, str):
        return ""
    values = list(tpr.values()) + list(fpr.values())
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        return ""
    w = _WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    keys = list(tpr)
    top = 1.2 * len(keys)
    panels = []
    for part, role, xlabel in ((tpr, "data", w["tpr"]), (fpr, "loss", w["fpr"])):
        p = chart.Panel(x=(0, 1), y=(0, top), xlabel=xlabel, yticks=[], xticks=[0, 0.5, 1],
                        xtick_labels=["0", "0.5", "1"])
        for i, k in enumerate(keys):
            y = top - 1.2 * (i + 1)
            p.text(0.005, y + 0.72, k + ": " + _num(part[k], sign), "ink", anchor="start", halo=True)   # Panel.text escapes it
            p.rect(0, y + 0.15, part[k], y + 0.45, role=role, solid=True)
        for x in (min(part.values()), max(part.values())):
            p.line([x, x], [0, top], role="ink-3", width=1, dashed=True)
        panels.append(p)

    def shown(part):
        return "; ".join(f"{k} {_num(part[k], sign)}" for k in keys)

    svg = chart.figure(panels, alt=w["alt"].format(column=column, tpr=shown(tpr), fpr=shown(fpr)), width=300,
                       height=max(160, 44 * len(keys) + 70))
    return yours.figure(w["title"].format(column=column), chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


_CLASH_WORDS = {
    "en": {"title": "Your false positive rates with the same precision and recall", "xlabel": "false positive rate",
           "base": "base rate {p}",
           "alt": "One bar per group, as long as its false positive rate: {shown}."},
    "nl": {"title": "Jouw false positive rates bij dezelfde precision en recall", "xlabel": "false positive rate",
           "base": "base rate {p}",
           "alt": "Eén balk per groep, zo lang als de false positive rate: {shown}."},
}


def _fprs(v) -> dict | None:
    """The learner's false positive rate per group, when it is a dict of 1 to 6 numbers from 0 to 1."""
    if not isinstance(v, dict) or not 1 <= len(v) <= 6:
        return None
    out = {}
    for k, x in v.items():
        x = _number(x)
        if not isinstance(k, str) or x is None or not math.isfinite(x) or not 0 <= x <= 1:
            return None
        out[k] = x
    return out


def _clash_picture(seen) -> str:
    """Per group a bar as long as the learner's false positive rate, with the group's base rate beside its name."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    value = _fprs(names.get("fpr"))
    p_of = names.get("p")
    if value is None or not isinstance(p_of, dict):
        return ""
    w = _CLASH_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    keys = list(value)
    top = 1.2 * len(keys)
    panel = chart.Panel(x=(0, 0.25), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25],
                        xtick_labels=["0", "0.05", "0.1", "0.15", "0.2", "0.25"])
    for i, k in enumerate(keys):
        y = top - 1.2 * (i + 1)
        base = _number(p_of.get(k))
        label = k + ": " + _num(value[k], sign)
        if base is not None:
            label += " (" + w["base"].format(p=f"{base:.3f}".replace(".", sign)) + ")"
        panel.text(0.001, y + 0.72, label, "ink", anchor="start", halo=True)   # Panel.text escapes it
        panel.rect(0, y + 0.15, min(value[k], 0.25), y + 0.45, role="loss", solid=True)
    shown = "; ".join(f"{k} {_num(value[k], sign)}" for k in keys)
    svg = chart.figure([panel], alt=w["alt"].format(shown=shown), width=520, height=max(150, 44 * len(keys) + 70))
    return yours.figure(w["title"], chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


def _clash_close(got, want) -> bool:
    """A false positive rate per group, the same groups, each the same to two decimals."""
    a, b = _fprs(got), _fprs(want)
    return a is not None and b is not None and _same(a, b)


def _clash_like(make):
    def match(got, text):
        if not isinstance(got, dict) or set(got) != set(P):
            return False
        want = make()
        return all(_number(got[k]) is not None and abs(got[k] - want[k]) < 0.002 for k in P)
    return match


def _misses():
    return {g: ((1 - RECALL) * 1000 * s / PRECISION - (1 - RECALL) * 1000 * s) / (1000 - 1000 * s) for g, s in P.items()}


def _over_positives():
    return {g: RECALL * (1 - PRECISION) / PRECISION for g in P}


def _over_all():
    return {g: (RECALL * 1000 * s / PRECISION - RECALL * 1000 * s) / 1000 for g, s in P.items()}


def _wrong_share():
    return {g: 1 - PRECISION for g in P}


def _tiny(got, text) -> bool:
    return (isinstance(got, dict) and set(got) == set(P)
            and all(_number(x) is not None and 0 < x < 0.001 for x in got.values()))


def _all_yes():
    return {g: RECALL * 1000 * s / PRECISION / (1000 - 1000 * s) for g, s in P.items()}


def _times():
    return {g: (RECALL * 1000 * s * PRECISION - RECALL * 1000 * s) / (1000 - 1000 * s) for g, s in P.items()}


def _percent(got, text) -> bool:
    return isinstance(got, dict) and any(_number(x) is not None and x > 1 for x in got.values())


def _one_group(got, text) -> bool:
    return isinstance(got, dict) and 1 <= len(got) < len(P) and set(got) <= set(P)


def _other_keys(got, text) -> bool:
    return isinstance(got, dict) and len(got) >= 1 and not set(got) <= set(P)


def _negative(got, text) -> bool:
    return isinstance(got, dict) and any(_number(x) is not None and x < 0 for x in got.values())


# The lesson is done when these are correct.
REQUIRED = ["errors", "clash", "which"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_READ_LINE = 'people = pd.read_csv("data/adult.csv.gz")'
_COLUMN_LINE = f'column = "{COLUMN}"'
_HEAD = ("import pandas as pd\n\n# 48,842 people from a US survey of 1994 (UCI Adult)\n" + _READ_LINE + "\n"
         + _COLUMN_LINE + "\n\n")
_START = _HEAD + "...\ntpr, fpr\n"
_BODY = ('rich = people[people["income"] == ">50K"]\n'
         f'tpr = rich[rich["education_num"] >= {SCHOOL}].groupby(column).size() / rich.groupby(column).size()\n'
         'poor = people[people["income"] == "<=50K"]\n'
         f'fpr = poor[poor["education_num"] >= {SCHOOL}].groupby(column).size() / poor.groupby(column).size()\n')
_SOLUTION = _HEAD + _BODY + "tpr, fpr"


def _s(body: str, last: str = "tpr, fpr") -> str:
    return _HEAD + body + last


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1, old
    return _s(_BODY.replace(old, new))


_P_LINE = 'p = {"women": 0.109, "men": 0.300}'
_PRECISION_LINE = f"precision = {PRECISION}"
_RECALL_LINE = f"recall = {RECALL}"
_CLASH_HEAD = ("# the base rates in the test set of UCI Adult: the share that earns more than 50,000 dollars\n"
               + _P_LINE + "\n# the same precision and the same recall in both groups\n" + _PRECISION_LINE + "\n"
               + _RECALL_LINE + "\n\nfpr = {}\nfor group in p:\n")
_CLASH_START = _CLASH_HEAD + "    ...\nfpr\n"
_CLASH_BODY = ("    positives = 1000 * p[group]\n"
               "    true_yes = recall * positives\n"
               "    false_yes = true_yes / precision - true_yes\n"
               "    fpr[group] = false_yes / (1000 - positives)\n")
_CLASH_SOLUTION = _CLASH_HEAD + _CLASH_BODY + "fpr"


def _c(body: str, last: str = "fpr") -> str:
    return _CLASH_HEAD + body + last


def _cswap(old: str, new: str) -> str:
    assert _CLASH_BODY.count(old) == 1, old
    return _c(_CLASH_BODY.replace(old, new))


_KEEP_TEXT = {"en": "Your code may be right, but the check needs the lines of `people` and `column` exactly as they were: put them back. The check tries another column by itself.",
              "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `column` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere kolom."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

EXERCISES = {
    "errors": CodeBox(
        name="errors",
        picture=_errors_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_READ_LINE, _COLUMN_LINE],
        compare=_close,
        digits=2,
        # The check also runs the box with another column.
        others=[{"column": OTHER_COLUMN}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_rich`. In the box the names have none: `people`, `column`, `rich`, `tpr`, `poor`, `fpr`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_rich`. In het vak hebben de namen dat niet: `people`, `column`, `rich`, `tpr`, `poor`, `fpr`."),
            Mistake("column", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks the table for a column it does not have. Group by the name `column` itself, without quotes: it holds the text `\"sex\"`. The income is in the column `income`, the years of schooling in `education_num`.",
                    nl="Je code vraagt de tabel om een kolom die hij niet heeft. Groepeer op de naam `column` zelf, zonder aanhalingstekens: daarin staat de tekst `\"sex\"`. Het inkomen staat in de kolom `income`, het aantal jaren school in `education_num`."),
            Mistake("text", error=lambda e: isinstance(e, TypeError),
                    en="Your code compares text with a number. Is the column `education_num`, the number of years, and not `education`, the words? And is `13` written without quotes?",
                    nl="Je code vergelijkt tekst met een getal. Is de kolom `education_num`, het aantal jaren, en niet `education`, de woorden? En staat `13` er zonder aanhalingstekens?"),
            Mistake("series", matches=_series,
                    en="Your last line gives one Series. The check wants two, with a comma between them: `tpr, fpr`.",
                    nl="Je laatste regel geeft één Series. De check wil er twee, met een komma ertussen: `tpr, fpr`."),
            Mistake("one-number", matches=_one_thing,
                    en="Your last line gives one number. The check wants the two Series, with a comma between them: `tpr, fpr`.",
                    nl="Je laatste regel geeft één getal. De check wil de twee Series, met een komma ertussen: `tpr, fpr`."),
            Mistake("missing", matches=_missing,
                    en="Some of your shares are `nan`: a selection has no rows. Are `rich` and `poor` the rows where `income` is exactly `\">50K\"` and `\"<=50K\"`?",
                    nl="Sommige van je delen zijn `nan`: een selectie heeft geen rijen. Zijn `rich` en `poor` de rijen waar `income` precies `\">50K\"` en `\"<=50K\"` is?"),
            Mistake("swapped", matches=_both_are(lambda: _want()[1], lambda: _want()[0]),
                    en="Your two Series are in the wrong order. The last line is `tpr, fpr`: first the recall, then the false positive rate.",
                    nl="Je twee Series staan in de verkeerde volgorde. De laatste regel is `tpr, fpr`: eerst de recall, dan de false positive rate."),
            Mistake("same", matches=_fpr_is(lambda: _want()[0]),
                    en="Your `fpr` is the same as your `tpr`: it still counts the people who earn more. For `poor`, take the rows where `income` is `\"<=50K\"`, and use `poor` on both sides of `fpr`.",
                    nl="Je `fpr` is gelijk aan je `tpr`: hij telt nog steeds de mensen die meer verdienen. Neem voor `poor` de rijen waar `income` gelijk is aan `\"<=50K\"`, en gebruik `poor` aan beide kanten van `fpr`."),
            Mistake("all", matches=_fpr_is(lambda: rule_rates(COLUMN, "all")),
                    en="Your `fpr` is the share of everyone in a group that gets a yes: the yes rate of the previous lesson. The false positive rate counts only the people who do not earn more: take the rows of `poor`.",
                    nl="Je `fpr` is het deel van iedereen in een groep dat een ja krijgt: het deel met een ja uit de vorige les. De false positive rate telt alleen de mensen die niet meer verdienen: neem de rijen van `poor`."),
            Mistake("of-yes", matches=_fpr_is(lambda: rule_rates(COLUMN, "poor", over="yes")),
                    en="You divided by all the people with a yes in a group: that is the share of the yes that is wrong, 1 minus the precision. The false positive rate divides by all the people who do not earn more: `poor.groupby(column).size()`.",
                    nl="Je deelde door alle mensen met een ja in een groep: dat is het deel van de ja's dat fout is, 1 min de precision. De false positive rate deelt door alle mensen die niet meer verdienen: `poor.groupby(column).size()`."),
            Mistake("no", matches=_fpr_is(lambda: rule_rates(COLUMN, "poor", "no")),
                    en="Your `fpr` counts the people who do not earn more and get no yes: the share the rule gets right. Count the ones with a yes: `education_num` of 13 or more.",
                    nl="Je `fpr` telt de mensen die niet meer verdienen en geen ja krijgen: het deel waar de regel gelijk heeft. Tel de mensen met een ja: een `education_num` van 13 of meer."),
            Mistake("complement", matches=_complement,
                    en="Your `fpr` is 1 minus the recall: the share of the people who earn more that gets no yes, the misses. The false positive rate looks at other people: those who do not earn more, in `poor`.",
                    nl="Je `fpr` is 1 min de recall: het deel van de mensen die meer verdienen dat geen ja krijgt, de gemiste gevallen. De false positive rate kijkt naar andere mensen: wie niet meer verdient, in `poor`."),
            Mistake("strict", matches=_both_are(lambda: rule_rates(COLUMN, "rich", "strict"),
                                                lambda: rule_rates(COLUMN, "poor", "strict")),
                    en="Your rule leaves out the people with exactly 13, the bachelor's degree itself. Use `>=`, not `>`.",
                    nl="Je regel laat de mensen met precies 13 weg, de bachelor zelf. Gebruik `>=`, niet `>`."),
            Mistake("counts", matches=_counts,
                    en="Some of your numbers are larger than 1, so they are counts, not shares. Divide the people with a yes per group by all the people of `rich` or `poor` per group.",
                    nl="Sommige van je getallen zijn groter dan 1, dus het zijn aantallen, geen delen. Deel de mensen met een ja per groep door alle mensen van `rich` of `poor` per groep."),
            Mistake("by-race", matches=_both_are(lambda: _want(OTHER_COLUMN)[0], lambda: _want(OTHER_COLUMN)[1]),
                    en="Your numbers are those per value of `race`, because you grouped by `race`, as in the worked example. Group by the column you are checking: `groupby(column)`.",
                    nl="Je getallen zijn die per waarde van `race`, omdat je op `race` groepeerde, zoals in het uitgewerkte voorbeeld. Groepeer op de kolom die je controleert: `groupby(column)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with four lines: `rich` and `tpr` as in the worked example, then `poor` and `fpr` in the same way, for the people who do not earn more.",
                      "nl": "Vervang de `...` door vier regels: `rich` en `tpr` zoals in het uitgewerkte voorbeeld, dan `poor` en `fpr` op dezelfde manier, voor de mensen die niet meer verdienen."},
            "cleared": _CLEARED,
            "keep": _KEEP_TEXT,
            "no_value": {"en": "The last line of the box shows nothing. Put `tpr, fpr` on the last line, so the check can see both.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `tpr, fpr` op de laatste regel, zodat de check ze allebei ziet."},
            "wrong": {"en": "Your numbers are not those of the rule yet. Is `tpr` the share of `rich` with an `education_num` of 13 or more, per group, and `fpr` the same share of `poor`, the rows where `income` is `\"<=50K\"`?",
                      "nl": "Je getallen zijn nog niet die van de regel. Is `tpr` het deel van `rich` met een `education_num` van 13 of meer, per groep, en `fpr` hetzelfde deel van `poor`, de rijen waar `income` gelijk is aan `\"<=50K\"`?"},
            "other": {"en": "Right for sex. But with {where}, your numbers are not right. Do you group by `column` itself, and not by the text `\"sex\"`?",
                      "nl": "Goed voor sex. Maar met {where} kloppen je getallen niet. Groepeer je op `column` zelf, en niet op de tekst `\"sex\"`?"},
            "other_error": {"en": "Right for sex. But with {where}, your code stops with an error. Do you use `column` and `people` themselves?",
                            "nl": "Goed voor sex. Maar met {where} stopt je code met een fout. Gebruik je `column` en `people` zelf?"},
            "correct": {"en": "The rule's recall is 0.53 for women and 0.49 for men, its false positive rate 0.18 and 0.16: by sex, its errors are almost equal, close to equalized odds. Yet a yes means something else per group: of the women with a yes, 0.26 earn more than 50,000 dollars, of the men 0.57. Exercise 2 shows why both cannot be equal here.",
                        "nl": "De recall van de regel is 0,53 voor vrouwen en 0,49 voor mannen, de false positive rate 0,18 en 0,16: per geslacht zijn de fouten bijna gelijk, dicht bij equalized odds. Toch betekent een ja per groep iets anders: van de vrouwen met een ja verdient 0,26 meer dan 50.000 dollar, van de mannen 0,57. Oefening 2 laat zien waarom hier niet allebei gelijk kan zijn."},
        },
    ),
    "clash": CodeBox(
        name="clash",
        picture=_clash_picture,
        start=_CLASH_START,
        solution=_CLASH_SOLUTION,
        keep=[_P_LINE, _PRECISION_LINE, _RECALL_LINE],
        compare=_clash_close,
        digits=2,
        # The check also runs the box with another precision and recall.
        others=[{"precision": 0.6, "recall": 0.7}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_positives`. In the box the names have none: `p`, `precision`, `recall`, `positives`, `true_yes`, `false_yes`, `fpr`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_positives`. In het vak hebben de namen dat niet: `p`, `precision`, `recall`, `positives`, `true_yes`, `false_yes`, `fpr`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks `p` for a key it does not have. Inside the loop, `group` holds the name of a group: write `p[group]`, without quotes around `group`.",
                    nl="Je code vraagt `p` om een sleutel die het niet heeft. In de loop bevat `group` de naam van een groep: schrijf `p[group]`, zonder aanhalingstekens om `group`."),
            Mistake("percent", matches=_percent,
                    en="A false positive rate above 1 is not a share. Did you divide by the number of people who do not earn more, `1000 - positives`?",
                    nl="Een false positive rate boven 1 is geen deel. Deelde je door het aantal mensen dat niet meer verdient, `1000 - positives`?"),
            Mistake("times", matches=_clash_like(_times),
                    en="You multiplied by the precision. The true yes are a share `precision` of all the yes, so all the yes are `true_yes / precision`: divide.",
                    nl="Je vermenigvuldigde met de precision. De terechte ja's zijn een deel `precision` van alle ja's, dus alle ja's zijn `true_yes / precision`: delen."),
            Mistake("negative", matches=_negative,
                    en="A false positive rate below 0 cannot be: you took the true yes minus all the yes. All the yes minus the true yes gives the false yes: `true_yes / precision - true_yes`.",
                    nl="Een false positive rate onder 0 kan niet: je nam de terechte ja's min alle ja's. Alle ja's min de terechte ja's geeft de onterechte: `true_yes / precision - true_yes`."),
            Mistake("one-group", matches=_one_group,
                    en="Your `fpr` has fewer groups than `p`. Does the line `fpr[group] = ...` stand inside the loop, indented by four spaces?",
                    nl="Je `fpr` heeft minder groepen dan `p`. Staat de regel `fpr[group] = ...` in de loop, vier spaties ingesprongen?"),
            Mistake("other-keys", matches=_other_keys,
                    en="Your `fpr` has other keys than `p`. Use `group` itself as the key, without quotes: `fpr[group] = ...`.",
                    nl="Je `fpr` heeft andere sleutels dan `p`. Gebruik `group` zelf als sleutel, zonder aanhalingstekens: `fpr[group] = ...`."),
            Mistake("misses", matches=_clash_like(_misses),
                    en="Your `true_yes` counts the people who earn more and get no yes: the share `1 - recall` of them. The true yes are the share `recall`: `true_yes = recall * positives`.",
                    nl="Je `true_yes` telt de mensen die meer verdienen en geen ja krijgen: het deel `1 - recall` daarvan. De terechte ja's zijn het deel `recall`: `true_yes = recall * positives`."),
            Mistake("over-positives", matches=_clash_like(_over_positives),
                    en="You divided the false yes by the people who earn more, so both groups come out the same. The false positive rate divides by the people who do not earn more: `1000 - positives`.",
                    nl="Je deelde de onterechte ja's door de mensen die meer verdienen, dus beide groepen komen hetzelfde uit. De false positive rate deelt door de mensen die niet meer verdienen: `1000 - positives`."),
            Mistake("over-all", matches=_clash_like(_over_all),
                    en="You divided the false yes by all 1,000 people. The false positive rate divides only by the people who do not earn more: `1000 - positives`.",
                    nl="Je deelde de onterechte ja's door alle 1000 mensen. De false positive rate deelt alleen door de mensen die niet meer verdienen: `1000 - positives`."),
            Mistake("wrong-share", matches=_clash_like(_wrong_share),
                    en="Your rates are 1 minus the precision: of all the yes, the share that is wrong. The false positive rate divides the false yes by the people who do not earn more: `1000 - positives`.",
                    nl="Je delen zijn 1 min de precision: van alle ja's het deel dat onterecht is. De false positive rate deelt de onterechte ja's door de mensen die niet meer verdienen: `1000 - positives`."),
            Mistake("tiny", matches=_tiny,
                    en="Your rates are tiny. Is `positives` a number of people out of 1,000, `1000 * p[group]`, and not the share itself?",
                    nl="Je delen zijn piepklein. Is `positives` een aantal mensen van de 1000, `1000 * p[group]`, en niet het deel zelf?"),
            Mistake("all-yes", matches=_clash_like(_all_yes),
                    en="Your `false_yes` counts all the yes, the true ones too. All the yes are `true_yes / precision`; take away the true yes: `true_yes / precision - true_yes`.",
                    nl="Je `false_yes` telt alle ja's, ook de terechte. Alle ja's zijn `true_yes / precision`; haal de terechte ja's eraf: `true_yes / precision - true_yes`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the loop with the four steps of *Why not both*, per 1,000 people of the group: `positives`, `true_yes`, `false_yes` and `fpr[group]`.",
                      "nl": "Vervang de `...` in de loop door de vier stappen van *Waarom niet allebei*, per 1000 mensen van de groep: `positives`, `true_yes`, `false_yes` en `fpr[group]`."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `p`, `precision` and `recall` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `p`, `precision` en `recall` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `fpr` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `fpr` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Per 1,000 people of a group: are `positives` the people who earn more, `true_yes` the share `recall` of them, `false_yes` all the yes (`true_yes / precision`) minus the true yes, and `fpr[group]` the false yes divided by the people who do not earn more?",
                      "nl": "Je laatste regel geeft `{got}`. Per 1000 mensen van een groep: zijn `positives` de mensen die meer verdienen, `true_yes` het deel `recall` daarvan, `false_yes` alle ja's (`true_yes / precision`) min de terechte, en `fpr[group]` de onterechte ja's gedeeld door de mensen die niet meer verdienen?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `precision` and `recall` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `precision` en `recall` zelf, en niet hun getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `p`, `precision` and `recall` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `p`, `precision` en `recall` zelf?"},
            "correct": {"en": "The women must have a false positive rate of 0.03 and the men of 0.11, more than three times as large. The same precision and the same recall leave no choice: the group where more people earn more gets more false yes. That is what Kleinberg and Chouldechova proved, and what the model of the demo shows.",
                        "nl": "De vrouwen moeten een false positive rate van 0,03 hebben en de mannen van 0,11, meer dan drie keer zo groot. Dezelfde precision en dezelfde recall laten geen keus: de groep waarin meer mensen meer verdienen, krijgt meer onterechte ja's. Dat bewezen Kleinberg en Chouldechova, en het model van de demo laat het zien."},
        },
    ),
    "which": Sort(
        name="which",
        groups={
            "yes": {"en": "The yes rates: demographic parity", "nl": "Hoe vaak elke groep ja krijgt: demographic parity"},
            "errors": {"en": "The errors: equalized odds", "nl": "De fouten: equalized odds"},
            "meaning": {"en": "What a score means: calibration within groups", "nl": "Wat een score betekent: calibration within groups"},
        },
        items={
            "limit": {"en": "**A threshold per group.** Women and men both get a yes 0.19 of the time.",
                      "nl": "**Een threshold per groep.** Vrouwen en mannen krijgen allebei in 0,19 van de gevallen een ja."},
            "interview": {"en": "**Interviews.** A firm invites the same share of every group to an interview.",
                          "nl": "**Gesprekken.** Een bedrijf nodigt van elke groep hetzelfde deel uit voor een gesprek."},
            "flagged": {"en": "**ProPublica.** Of the defendants who were not arrested again, the score flagged Black defendants almost twice as often as White defendants.",
                        "nl": "**ProPublica.** Van de verdachten die niet opnieuw werden opgepakt, markeerde de score zwarte verdachten bijna twee keer zo vaak als witte."},
            "missed": {"en": "**Missed.** Of the women who really earn more, the model gives 0.44 a yes; of such men, 0.57.",
                       "nl": "**Gemist.** Van de vrouwen die echt meer verdienen, geeft het model 0,44 een ja; van zulke mannen 0,57."},
            "seventy": {"en": "**About 0.7.** Of the people with a probability near 0.7, about 70 percent earn more, women and men alike.",
                        "nl": "**Ongeveer 0,7.** Van de mensen met een kans rond 0,7 verdient ongeveer 70 procent meer, bij vrouwen en mannen."},
            "northpointe": {"en": "**Northpointe.** Of the defendants the score flagged, about the same share was arrested again in each group.",
                            "nl": "**Northpointe.** Van de verdachten die de score markeerde, werd in elke groep ongeveer hetzelfde deel opnieuw opgepakt."},
        },
        solution={"limit": "yes", "interview": "yes", "flagged": "errors", "missed": "errors", "seventy": "meaning",
                  "northpointe": "meaning"},
        nudge={
            "limit": {"en": "Does the claim look at whether the yes were right, or only at how often each group gets one?",
                      "nl": "Kijkt de bewering of de ja's terecht waren, of alleen hoe vaak elke groep er een krijgt?"},
            "interview": {"en": "The firm counts the invitations per group. Does it look at who would have been a good hire?",
                          "nl": "Het bedrijf telt de uitnodigingen per groep. Kijkt het naar wie een goede kracht zou zijn?"},
            "flagged": {"en": "The claim starts from the people who were not arrested again. What does it count among them?",
                        "nl": "De bewering begint bij de mensen die niet opnieuw werden opgepakt. Wat telt ze onder hen?"},
            "missed": {"en": "The claim starts from the people who really earn more: the share with a yes among them is the recall.",
                       "nl": "De bewering begint bij de mensen die echt meer verdienen: het deel met een ja onder hen is de recall."},
            "seventy": {"en": "The claim starts from a probability, and asks how often it comes true in each group.",
                        "nl": "De bewering begint bij een kans, en vraagt hoe vaak die in elke groep uitkomt."},
            "northpointe": {"en": "The claim starts from the people with a flag: the share of them that was arrested again is the precision.",
                            "nl": "De bewering begint bij de mensen met een markering: het deel van hen dat opnieuw werd opgepakt, is de precision."},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the claims again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de beweringen opnieuw uit."},
            "empty": {"en": "Sort each claim by what it compares between the groups.",
                      "nl": "Deel elke bewering in bij wat ze tussen de groepen vergelijkt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The claim *{item}* belongs somewhere else.",
                      "nl": "De bewering *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. Where a claim starts tells you what it measures: from everyone, it counts the yes; from the people with or without the outcome, it counts the errors; from the people with a yes or a score, it asks what the score means. ProPublica and Northpointe started from different people.",
                        "nl": "Alle {n} kloppen. Waar een bewering begint, vertelt wat ze meet: bij iedereen telt ze de ja's; bij de mensen met of zonder de uitkomst telt ze de fouten; bij de mensen met een ja of een score vraagt ze wat de score betekent. ProPublica en Northpointe begonnen bij andere mensen."},
        },
    ),
}

_SORTED = {"limit": "yes", "interview": "yes", "flagged": "errors", "missed": "errors", "seventy": "meaning",
           "northpointe": "meaning"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("errors", _START, "empty", None),
    ("errors", _SOLUTION, "correct", None),
    ("errors", _s(_BODY.replace("rich.groupby(column).size()\n", "rich.groupby(column).size()\ntpr = tpr.round(2)\n")
                  .replace("poor.groupby(column).size()\n", "poor.groupby(column).size()\nfpr = fpr.round(2)\n")),
     "correct", None),
    ("errors", _s('rich = people[people["income"] == ">50K"]\npoor = people[people["income"] != ">50K"]\n'
                  'rich_yes = rich[rich["education_num"] > 12]\npoor_yes = poor[poor["education_num"] > 12]\n'
                  "tpr = rich_yes.groupby(column).size() / rich.groupby(column).size()\n"
                  "fpr = poor_yes.groupby(column).size() / poor.groupby(column).size()\n"), "correct", None),
    ("errors", _swap("fpr = poor[", "fpr = _poor["), "error", "underscore"),
    ("errors", _swap("rich.groupby(column).size()\n", 'rich.groupby("column").size()\n'), "error", "column"),
    ("errors", _swap('rich["education_num"] >= 13', 'rich["education_num"] >= "13"'), "error", "text"),
    ("errors", _swap('people[people["income"] == ">50K"]', 'people[people["incme"] == ">50K"]'), "error", "column"),
    ("errors", _s(_BODY, "fpr"), "wrong", "series"),
    ("errors", _s(_BODY, "fpr.max() - fpr.min()"), "wrong", "one-number"),
    ("errors", _swap('people["income"] == "<=50K"', 'people["income"] == "<= 50K"'), "wrong", "missing"),
    ("errors", _s(_BODY, "fpr, tpr"), "wrong", "swapped"),
    ("errors", _swap('poor = people[people["income"] == "<=50K"]', 'poor = people[people["income"] == ">50K"]'),
     "wrong", "same"),
    ("errors", _swap('poor = people[people["income"] == "<=50K"]', "poor = people"), "wrong", "all"),
    ("errors", _swap("/ poor.groupby(column).size()", '/ people[people["education_num"] >= 13].groupby(column).size()'),
     "wrong", "of-yes"),
    ("errors", _swap('poor[poor["education_num"] >= 13]', 'poor[poor["education_num"] < 13]'), "wrong", "no"),
    ("errors", _s(_BODY + "fpr = 1 - tpr\n"), "wrong", "complement"),
    ("errors", _s(_BODY.replace(">= 13", "> 13")), "wrong", "strict"),
    ("errors", _swap(" / poor.groupby(column).size()", ""), "wrong", "counts"),
    ("errors", _s(_BODY.replace("groupby(column)", 'groupby("race")')), "wrong", "by-race"),
    ("errors", _s(_BODY.replace("groupby(column)", 'groupby("relationship")')), "wrong", None),
    ("errors", _s(_BODY.replace("groupby(column)", 'groupby("sex")')), "almost", "other-value"),
    ("errors", _SOLUTION.replace(_COLUMN_LINE, 'column = "race"'), "wrong", "changed-line"),
    ("errors", _s("...\ntpr = 0.5\nfpr = 0.2\n"), "wrong", "dots-left"),
    ("clash", _CLASH_START, "empty", None),
    ("clash", _CLASH_SOLUTION, "correct", None),
    ("clash", _c("    positives = 1000 * p[group]\n    true_yes = recall * positives\n"
                 "    false_yes = true_yes / precision - true_yes\n"
                 "    fpr[group] = round(false_yes / (1000 - positives), 2)\n"), "correct", None),
    ("clash", _c("    share = p[group]\n    fpr[group] = share / (1 - share) * (1 - precision) / precision * recall\n"),
     "correct", None),
    ("clash", _cswap("    true_yes = recall * positives", "    true_yes = recall * _positives"), "error", "underscore"),
    ("clash", _cswap("1000 * p[group]", '1000 * p["group"]'), "error", "key"),
    ("clash", _cswap("false_yes / (1000 - positives)", "100 * false_yes / (1000 - positives)"), "wrong", "percent"),
    ("clash", _cswap("true_yes / precision - true_yes", "true_yes - true_yes / precision"), "wrong", "negative"),
    ("clash", _c(_CLASH_BODY.replace("    fpr[group] = false_yes / (1000 - positives)\n", "")
                 + "fpr[group] = false_yes / (1000 - positives)\n"), "wrong", "one-group"),
    ("clash", _cswap("fpr[group] =", 'fpr["group"] ='), "wrong", "other-keys"),
    ("clash", _cswap("fpr[group] =", "fpr[group.upper()] ="), "wrong", "other-keys"),
    ("clash", _cswap("false_yes / (1000 - positives)", "false_yes / positives"), "wrong", "over-positives"),
    ("clash", _cswap("false_yes / (1000 - positives)", "false_yes / 1000"), "wrong", "over-all"),
    ("clash", _cswap("true_yes / precision - true_yes", "true_yes / precision"), "wrong", "all-yes"),
    ("clash", _cswap("true_yes / precision - true_yes", "true_yes * precision - true_yes"), "wrong", "times"),
    ("clash", _cswap("    true_yes = recall * positives", "    true_yes = (1 - recall) * positives"), "wrong", "misses"),
    ("clash", _cswap("    positives = 1000 * p[group]", "    positives = p[group]"), "wrong", "tiny"),
    ("clash", _cswap("false_yes / (1000 - positives)", "false_yes / (true_yes / precision)"), "wrong", "wrong-share"),
    ("clash", _cswap("    positives = 1000 * p[group]", "    positives = 100 * p[group]"), "wrong", None),
    ("clash", _c(_CLASH_BODY.replace("recall *", "0.57 *").replace("/ precision", "/ 0.69")), "almost",
     "other-value"),
    ("clash", _CLASH_SOLUTION.replace(_PRECISION_LINE, "precision = 0.8"), "wrong", "changed-line"),
    ("clash", _c("    ...\n    fpr[group] = 0.1\n"), "wrong", "dots-left"),
    ("which", {}, "empty", None),
    ("which", _SORTED, "correct", None),
    ("which", {"limit": "yes", "seventy": "meaning"}, "wrong", "unfinished"),
    ("which", {**_SORTED, "northpointe": "errors"}, "wrong", "wrong:northpointe"),
    ("which", {**_SORTED, "flagged": "meaning"}, "wrong", "wrong:flagged"),
    ("which", {**_SORTED, "missed": "yes"}, "wrong", "wrong:missed"),
]
