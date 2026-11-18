# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-global-surrogate: the random forest of rai-pdp-ice on three columns of UCI Adult, copied
into one small decision tree that learns the forest's answers on the training half: a global surrogate. The learner
writes fidelity, how often two lists of answers agree; finds the one question about education that agrees most with
the forest; works out how far the copy's accuracy can be from the forest's; and says what high fidelity does not show.
With the slider over the depth of the copy, for how many people of each education level it disagrees with the forest;
under the exercises, the learner's fidelity for the copies of depth 1 to 8, and the learner's scores of each question."""
import functools
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
DEPTHS = list(range(1, 9))       # the stops of the slider, and the copies the picture of exercise 1 measures
EDUCATION = list(range(1, 17))   # the values of education_num


@functools.lru_cache(maxsize=1)
def table():
    """The three columns of UCI Adult as floats, who earns more than 50,000 dollars, and the positions of the training
    half and of the test half (the table shuffled with seed 0 and cut in two). Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    X = people[COLUMNS].to_numpy().astype(float)
    y = (people["income"] == ">50K").to_numpy()
    order = np.random.default_rng(0).permutation(len(y))
    train, test = order[: len(y) // 2], order[len(y) // 2:]
    for a in (X, y, train, test):
        a.flags.writeable = False
    return X, y, train, test


@functools.lru_cache(maxsize=1)
def forest():
    """The forest of the worked example: 100 trees whose leaves hold at least 20 people, grown on the training half.
    About half a second."""
    from sklearn.ensemble import RandomForestClassifier
    X, y, train, test = table()
    return RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(X[train], y[train])


@functools.lru_cache(maxsize=1)
def answers() -> tuple:
    """What the forest says for each person of the training half and of the test half: True for a yes. Read-only."""
    X, y, train, test = table()
    out = forest().predict(X[train]), forest().predict(X[test])
    for a in out:
        a.flags.writeable = False
    return out


@functools.lru_cache(maxsize=None)
def copy(depth: int | None):
    """A tree of this depth grown on the forest's answers on the training half: a global surrogate of the forest.
    None grows it until every part is pure."""
    from sklearn.tree import DecisionTreeClassifier
    X, y, train, test = table()
    return DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[train], answers()[0])


@functools.lru_cache(maxsize=None)
def copy_says(depth: int | None):
    """What the copy of this depth says for each person of the test half. Read-only."""
    X, y, train, test = table()
    out = copy(depth).predict(X[test])
    out.flags.writeable = False
    return out


def agree(a, b) -> float:
    """The share of places where two arrays of answers are the same."""
    import numpy as np
    return float((np.asarray(a) == np.asarray(b)).mean())


def per_level(says, *, half: str = "test") -> list:
    """For each education_num from 1 to 16: (people, the copy says yes and the forest no, the copy no and the forest
    yes), for the people of this half."""
    X, y, train, test = table()
    rows = X[test] if half == "test" else X[train]
    forest_says = answers()[1] if half == "test" else answers()[0]
    out = []
    for e in EDUCATION:
        at = rows[:, 1] == e
        out.append((int(at.sum()), int((says[at] & ~forest_says[at]).sum()), int((~says[at] & forest_says[at]).sum())))
    return out


def by_degree(says) -> tuple:
    """Fidelity on the test half for people with education_num up to 12 and with 13 or more, and how many each are."""
    X, y, train, test = table()
    degree = X[test][:, 1] >= 13
    forest_says = answers()[1]
    return (agree(says[~degree], forest_says[~degree]), int((~degree).sum()),
            agree(says[degree], forest_says[degree]), int(degree.sum()))


def _show(x: float, lang: str, digits: int = 3) -> str:
    """A number with this many decimals and a real minus sign; a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _percent(x: float, lang: str) -> str:
    return _show(100 * x, lang, 1) + "%"


def _count(n: int, lang: str) -> str:
    """A whole number, with a separator from five digits: 18,374 and in Dutch 18.374; 6047 stays 6047."""
    if abs(n) < 10000:
        return str(n)
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

# --- the slider of *The idea*: where the copy disagrees with the forest ---------------------------------------------

_DEPTH_WORDS = {
    "en": {"title": "where the copy disagrees with the forest", "x": "education_num", "y": "people",
           "alt": "Bars for each value of education_num from 1 to 16: for how many people of the test half the copy of "
                  "depth {d} disagrees with the forest. Filled: the copy says yes and the forest no. Open: the copy says "
                  "no and the forest yes. The copy agrees with the forest on {all} of the test half: {low} of the people "
                  "with education up to 12, {high} of those with 13 or more.",
           "said": "Depth {d}, {leaves} leaves: the copy agrees with the forest on "
                   "<span class=\"c-pred ea-num\">{all}</span> of the test half; on {low} of the {nlow} people with "
                   "education up to 12, and on {high} of the {nhigh} with 13 or more. A copy that says no to everyone "
                   "agrees on {base}."},
    "nl": {"title": "waar de kopie het oneens is met het forest", "x": "education_num", "y": "mensen",
           "alt": "Staven voor elke waarde van education_num van 1 tot 16: voor hoeveel mensen van de testhelft de kopie "
                  "met diepte {d} het oneens is met het forest. Gevuld: de kopie zegt ja en het forest nee. Open: de kopie "
                  "zegt nee en het forest ja. De kopie is het met het forest eens bij {all} van de testhelft: bij {low} "
                  "van de mensen met een opleiding tot 12, bij {high} van wie 13 of meer heeft.",
           "said": "Diepte {d}, {leaves} bladeren: de kopie is het met het forest eens bij "
                   "<span class=\"c-pred ea-num\">{all}</span> van de testhelft; bij {low} van de {nlow} mensen met een "
                   "opleiding tot 12, en bij {high} van de {nhigh} met 13 of meer. Een kopie die tegen iedereen nee "
                   "zegt, is het eens bij {base}."},
}


@functools.lru_cache(maxsize=1)
def _tallest() -> int:
    """The highest bar over all depths of the slider, so that every depth is drawn on the same scale."""
    top = 0
    for d in DEPTHS:
        for n, a, b in per_level(copy_says(d)):
            top = max(top, a + b)
    return top


def baseline() -> float:
    """How often a copy that says no to everyone agrees with the forest on the test half."""
    return agree(answers()[1], False)


def depth_picture(d: int, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: for each education level, for how many people of the test half the copy of depth
    d disagrees with the forest, in two kinds; and the sentence with its fidelity, overall and in two groups."""
    w = _DEPTH_WORDS[lang]
    d = int(d)
    says = copy_says(d)
    top = _tallest() * 1.08
    p = chart.Panel(x=(0.4, 16.6), y=(0, top), title=w["title"], xlabel=w["x"], ylabel=w["y"],
                    xticks=[1, 4, 8, 12, 16])
    for e, (n, a, b) in zip(EDUCATION, per_level(says)):
        if a:
            p.rect(e - 0.36, 0, e + 0.36, a, role="loss", solid=True)
        if b:
            p.rect(e - 0.36, a, e + 0.36, a + b, role="loss")
    low, nlow, high, nhigh = by_degree(says)
    fill = {"d": d, "leaves": copy(d).get_n_leaves(), "all": _percent(agree(says, answers()[1]), lang),
            "low": _percent(low, lang), "nlow": _count(nlow, lang), "high": _percent(high, lang),
            "nhigh": _count(nhigh, lang), "base": _percent(baseline(), lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=460, height=270)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, w["said"].format(**fill)


# --- exercise 1: fidelity --------------------------------------------------------------------------------------------

def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _number(v):
    """A float from what a function gave back, or None when it is not one number."""
    if isinstance(v, bool):
        return None
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        v = v.item()
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _numbers(got, n: int):
    """A list of n floats from what the learner's code gave, or None."""
    if not isinstance(got, (list, tuple)) or len(got) != n:
        return None
    out = [_number(v) for v in got]
    return None if any(v is None for v in out) else out


_FID_WORDS = {
    "en": {"title": "Your fidelity(), for the copies of depth 1 to 8", "x": "depth of the copy", "y": "your fidelity()",
           "no": "says no to everyone",
           "alt": "Dots joined by a line: what your fidelity() gives for the answers of the copy of each depth from 1 to "
                  "8 and of the forest, for the 24,421 people of the test half. A dashed line: what it gives for a copy "
                  "that says no to everyone. {key}",
           "key": "Your fidelity() gives {v1} for the copy of depth 1, {v3} for depth 3 and {v8} for depth 8, and {base} "
                  "for a copy that says no to everyone."},
    "nl": {"title": "Jouw fidelity(), voor de kopieën met diepte 1 tot 8", "x": "diepte van de kopie",
           "y": "jouw fidelity()", "no": "zegt tegen iedereen nee",
           "alt": "Punten met een lijn ertussen: wat je fidelity() geeft voor de antwoorden van de kopie met elke diepte "
                  "van 1 tot 8 en van het forest, voor de 24.421 mensen van de testhelft. Een gestippelde lijn: wat hij "
                  "geeft voor een kopie die tegen iedereen nee zegt. {key}",
           "key": "Je fidelity() geeft {v1} voor de kopie met diepte 1, {v3} voor diepte 3 en {v8} voor diepte 8, en "
                  "{base} voor een kopie die tegen iedereen nee zegt."},
}


@functools.lru_cache(maxsize=1)
def _answer_lists() -> tuple:
    """The answers the picture of exercise 1 hands the learner's fidelity(), as lists of True and False: the forest's,
    those of each copy in DEPTHS, and a copy that says no to everyone, all on the test half."""
    forest_says = [bool(v) for v in answers()[1].tolist()]
    copies = tuple(tuple(bool(v) for v in copy_says(d).tolist()) for d in DEPTHS)
    return tuple(forest_says), copies, tuple([False] * len(forest_says))


def _fid_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    forest_says, copies, nobody = _answer_lists()

    def run():
        out = []
        for says in copies + (nobody,):
            v = _number(fn(list(says), list(forest_says)))
            if v is None or not 0 <= v <= 1:
                return None
            out.append(v)
        return out

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    lang = seen.lang
    w = _FID_WORDS[lang]
    values, base = got[:-1], got[-1]
    fill = {"v1": _show(values[0], lang), "v3": _show(values[2], lang), "v8": _show(values[7], lang),
            "base": _show(base, lang)}
    key = w["key"].format(**fill)
    low = max(0.0, math.floor((min(got) - 0.03) * 20) / 20)
    p = chart.Panel(x=(0.6, 8.4), y=(low, 1), title="", xlabel=w["x"], ylabel=w["y"], xticks=DEPTHS)
    p.line([0.6, 8.4], [base, base], role="ink-3", width=1.6, dashed=True)
    p.text(0.75, base + (1 - low) * 0.04, w["no"], role="ink-2", anchor="start", halo=True)
    p.line(DEPTHS, values, role="pred", width=2.4)
    p.points(DEPTHS, values, role="pred", r=4.5)
    svg = chart.figure([p], alt=w["alt"].format(key=key), width=520, height=260)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- exercise 2: the best question about education ------------------------------------------------------------------

def _counts(half: str) -> tuple:
    """For each education_num from 1 to 16 in this half: how many people, how many the forest says yes to, how many
    truly earn more."""
    X, y, train, test = table()
    rows, says, truth = (X[train], answers()[0], y[train]) if half == "train" else (X[test], answers()[1], y[test])
    people, forest_yes, true_yes = [], [], []
    for e in EDUCATION:
        at = rows[:, 1] == e
        people.append(int(at.sum()))
        forest_yes.append(int(says[at].sum()))
        true_yes.append(int(truth[at].sum()))
    return people, forest_yes, true_yes


# The training half as the box gives it (tests/test_rai_global_surrogate.py counts it again).
PEOPLE = [42, 125, 273, 495, 387, 687, 930, 323, 7764, 5506, 1029, 797, 4016, 1328, 421, 298]
FOREST_YES = [0, 0, 0, 0, 0, 0, 0, 0, 42, 102, 26, 61, 1832, 911, 368, 255]
TRUE_YES = [1, 5, 12, 28, 22, 36, 46, 22, 1210, 1046, 255, 216, 1661, 707, 315, 208]
THRESHOLDS = list(range(1, 18))   # t: the copy says yes when education_num >= t; at 17 it says no to everyone


def scores_of(people: list, yes: list) -> list:
    """For each t in THRESHOLDS, the share of people for whom "yes when education_num >= t" says what `yes` counts."""
    out = []
    for t in THRESHOLDS:
        same = 0
        for e in range(16):
            same += yes[e] if e + 1 >= t else people[e] - yes[e]
        out.append(same / sum(people))
    return out


_STUMP_WORDS = {
    "en": {"title": "Your scores, beside the forest's answers on the training half", "x1": "education_num",
           "y1": "people", "x2": "t", "y2": "your score", "best": "your best: t = {b}",
           "line": "; a dashed line just before education_num {b}, the t of your highest score",
           "alt": "Left: for each value of education_num from 1 to 16, the people of the training half, filled where the "
                  "forest says yes, open where it says no{line}. Right: dots joined by a line, your score for each t from 1 to 17. {key}",
           "key": "Your scores are highest at t = {b}, with {v}. Your score at t = 17, where the question says no to "
                  "everyone, is {last}."},
    "nl": {"title": "Jouw scores, naast de antwoorden van het forest op de helft om op te trainen", "x1": "education_num",
           "y1": "mensen", "x2": "t", "y2": "jouw score", "best": "jouw beste: t = {b}",
           "line": "; een gestippelde lijn net vóór education_num {b}, de t van je hoogste score",
           "alt": "Links: voor elke waarde van education_num van 1 tot 16 de mensen van de helft om op te trainen, gevuld "
                  "waar het forest ja zegt, open waar het nee zegt{line}. Rechts: punten met een lijn ertussen, je score voor elke t van 1 tot 17. {key}",
           "key": "Je scores zijn het hoogst bij t = {b}, met {v}. Je score bij t = 17, waar de vraag tegen iedereen nee "
                  "zegt, is {last}."},
}


def _stump_picture(seen) -> str:
    mine = seen.mine
    if mine is None:
        return ""
    got = _numbers(mine.value, len(THRESHOLDS))
    if got is None or any(not 0 <= v <= 1 for v in got):
        return ""
    lang = seen.lang
    w = _STUMP_WORDS[lang]
    b = THRESHOLDS[max(range(len(got)), key=lambda i: got[i])]
    fill = {"b": b, "v": _show(max(got), lang), "last": _show(got[-1], lang)}
    key = w["key"].format(**fill)
    alt = w["alt"].format(key=key, line=w["line"].format(b=b) if b <= 16 else "")
    left = chart.Panel(x=(0.4, 16.6), y=(0, max(PEOPLE) * 1.08), title="", xlabel=w["x1"], ylabel=w["y1"],
                       xticks=[1, 4, 8, 12, 16])
    for e, n, yes in zip(EDUCATION, PEOPLE, FOREST_YES):
        if yes:
            left.rect(e - 0.36, 0, e + 0.36, yes, role="pred", solid=True)
        left.rect(e - 0.36, yes, e + 0.36, n, role="ink-3")
    if b <= 16:
        left.line([b - 0.5, b - 0.5], [0, max(PEOPLE) * 1.04], role="ink-2", width=1.6, dashed=True)
    low = max(0.0, math.floor((min(got) - 0.03) * 20) / 20)
    right = chart.Panel(x=(0.4, 17.6), y=(low, 1), title="", xlabel=w["x2"], ylabel=w["y2"], xticks=[1, 5, 9, 13, 17])
    right.line(THRESHOLDS, got, role="pred", width=2.2)
    right.points(THRESHOLDS, got, role="pred", r=3.6)
    right.text(b, max(got) + (1 - low) * 0.06, w["best"].format(b=b), role="ink-2", halo=True)
    svg = chart.figure([left, right], alt=alt, width=600, height=260)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "between" and "reasons" are choice questions.
REQUIRED = ["fidelity", "stump", "between", "reasons"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_F_HEAD = ("def fidelity(copy, original):\n"
           "    # copy, original: lists of True and False of the same length, one answer per person\n"
           "    # the share of places where copy gives the same answer as original\n")
_F_START = _F_HEAD + "    ...\n"


def _f(cond: str = "        if copy[i] == original[i]:\n", ret: str = "    return same / len(copy)\n") -> str:
    """The body of fidelity with one part changed: the question per place, or the return."""
    return _F_HEAD + "    same = 0\n    for i in range(len(copy)):\n" + cond + "            same = same + 1\n" + ret


_F_SOLUTION = _f()
_F_COUNT = _f(ret="    return same\n")
_F_DIFFER = _f(cond="        if copy[i] != original[i]:\n")
_F_YES = _f(cond="        if copy[i] and original[i]:\n")
_F_WHOLE = _F_HEAD + "    return copy == original\n"
_F_FIRST = _f(ret="        return same / len(copy)\n")
_F_VISIBLE = [([True, False, False, True], [True, False, True, True])]
_F_HIDDEN = [
    ([False, False, False], [False, False, False]),
    ([True, True], [False, False]),
    ([True, False, True, False, False], [True, True, True, False, False]),
    ([False, True, False, False, True, False, False, False, False, False],
     [False, False, False, False, True, False, False, False, True, False]),
    ([True], [True]),
]

# --- exercise 2
_S_HEAD = ("# the training half, by education_num from 1 to 16: how many people have it, how many of them the forest\n"
           "# says yes to, and how many of them truly earn more than 50,000 dollars\n"
           f"people = {PEOPLE}\n"
           f"forest_yes = {FOREST_YES}\n"
           f"true_yes = {TRUE_YES}\n"
           "\n"
           "# for each t from 1 to 17: how often the copy \"yes when education_num >= t\" agrees with the forest\n"
           "scores = []\n"
           "for t in range(1, 18):\n")
_S_START = _S_HEAD + "    ...\nscores\n"


def _s(start: str = "    same = 0\n", cond: str = "        if e + 1 >= t:\n",
       yes: str = "            same = same + forest_yes[e]\n",
       no: str = "        else:\n            same = same + people[e] - forest_yes[e]\n",
       add: str = "    scores.append(same / sum(people))\n", before: str = "") -> str:
    """The loop over t with one part changed: where the count starts, the question, what a yes or a no adds, what is
    added to scores, and what comes before the loop."""
    head = _S_HEAD.replace("scores = []\n", "scores = []\n" + before)
    return head + start + "    for e in range(16):\n" + cond + yes + no + add + "scores\n"


_S_SOLUTION = _s()
_S_LABELS = _s(yes="            same = same + true_yes[e]\n",
               no="        else:\n            same = same + people[e] - true_yes[e]\n")
_S_FLIPPED = _s(cond="        if e + 1 < t:\n")
_S_SHIFTED = _s(cond="        if e >= t:\n")
_S_COUNT = _s(add="    scores.append(same)\n")
_S_YES_ONLY = _s(no="")
_S_NO_ONLY = _s(cond="        if e + 1 < t:\n", yes="            same = same + people[e] - forest_yes[e]\n", no="")
_S_RUNNING = _s(start="", before="same = 0\n")
_S_INSIDE = _s(add="").replace("same = same + people[e] - forest_yes[e]\n",
                               "same = same + people[e] - forest_yes[e]\n        scores.append(same / sum(people))\n")

EXERCISES = {
    "fidelity": FunctionBox(
        name="fidelity",
        picture=_fid_picture,
        function="fidelity",
        start=_F_START,
        solution=_F_SOLUTION,
        visible=_F_VISIBLE,
        hidden=_F_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("count", wrong=_F_COUNT,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the number of places where the two lists agree, not the share. Divide by the number of places, `len(copy)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft het aantal plaatsen waar de twee lijsten het eens zijn, niet het deel. Deel door het aantal plaatsen, `len(copy)`."),
            Mistake("differ", wrong=_F_DIFFER,
                    en="`{call}` gives {got}, and it should give {want}: the share of places where the two lists differ. Fidelity counts the places where they give the same answer: `==`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de plaatsen waar de twee lijsten verschillen. Fidelity telt de plaatsen waar ze hetzelfde antwoord geven: `==`."),
            Mistake("yes-only", wrong=_F_YES,
                    en="`{call}` gives {got}, and it should give {want}. You count only the places where both say `True`. Where both say `False` they agree too, and in UCI Adult that is most people.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt alleen de plaatsen waar allebei `True` zeggen. Waar allebei `False` zeggen zijn ze het ook eens, en in UCI Adult zijn dat de meeste mensen."),
            Mistake("whole", wrong=_F_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}. `copy == original` says whether the two whole lists are equal: one `True` or `False`. Fidelity compares them place by place, with a loop over `i`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `copy == original` zegt of de twee hele lijsten gelijk zijn: één `True` of `False`. Fidelity vergelijkt ze plaats voor plaats, met een loop over `i`."),
            Mistake("first", wrong=_F_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first place. If your `return` stands inside the `for` loop, Python leaves the function there. Put it after the loop, as far in as `same = 0`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na de eerste plaats. Als je `return` binnen de `for`-loop staat, verlaat Python de functie daar. Zet hem na de loop, even ver ingesprongen als `same = 0`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the body of `fidelity`: count the places `i` where `copy[i]` and `original[i]` are the same, and return that count divided by the number of places.",
                      "nl": "Vervang de `...` door de inhoud van `fidelity`: tel de plaatsen `i` waar `copy[i]` en `original[i]` gelijk zijn, en geef dat aantal gedeeld door het aantal plaatsen terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work it out by hand: at how many places do the two lists say the same, and how many places are there?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het met de hand uit: op hoeveel plaatsen zeggen de twee lijsten hetzelfde, en hoeveel plaatsen zijn er?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for lists of any length, and does it use `copy` and `original` as they are given?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor lijsten van elke lengte, en gebruikt hij `copy` en `original` zoals ze worden meegegeven?"},
            "correct": {"en": "Your `fidelity` is right in all {n} cases. The picture runs it on the answers of the forest and of the copies of depth 1 to 8 for the 24,421 people of the test half: from 0.881 at depth 1 to 0.991 at depth 8, while a copy that says no to everyone already gets 0.852.",
                        "nl": "Je `fidelity` klopt in alle {n} gevallen. De tekening voert hem uit op de antwoorden van het forest en van de kopieën met diepte 1 tot 8 voor de 24.421 mensen van de testhelft: van 0,881 bij diepte 1 tot 0,991 bij diepte 8, terwijl een kopie die tegen iedereen nee zegt al 0,852 haalt."},
        },
    ),
    "stump": CodeBox(
        name="stump",
        picture=_stump_picture,
        start=_S_START,
        digits=3,
        solution=_S_SOLUTION,
        first=[_DOTS],
        mistakes=[
            Mistake("labels", wrong=_S_LABELS,
                    en="Your scores say how often the copy matches who truly earns more: its accuracy, from `true_yes`. A copy learns from the forest's answers, so count with `forest_yes`.",
                    nl="Je scores zeggen hoe vaak de kopie klopt met wie echt meer verdient: zijn accuracy, uit `true_yes`. Een kopie leert van de antwoorden van het forest, dus tel met `forest_yes`."),
            Mistake("flipped", wrong=_S_FLIPPED,
                    en="Your scores are 1 minus the right ones: your copy says yes below `t` instead of from `t` on. It says yes when education_num is at least `t`: `e + 1 >= t`.",
                    nl="Je scores zijn 1 min de goede: je kopie zegt ja onder `t` in plaats van vanaf `t`. Hij zegt ja als education_num minstens `t` is: `e + 1 >= t`."),
            Mistake("shifted", wrong=_S_SHIFTED,
                    en="Your scores are shifted by one: your copy says yes from education_num `t + 1` on, one level late. Level `e` is education_num `e + 1`: ask `e + 1 >= t`.",
                    nl="Je scores zijn één plaats opgeschoven: je kopie zegt ja vanaf education_num `t + 1`, één niveau te laat. Niveau `e` is education_num `e + 1`: vraag `e + 1 >= t`."),
            Mistake("count", wrong=_S_COUNT,
                    en="Your scores are numbers of people, not shares. Divide each by the number of people in the half, `sum(people)`.",
                    nl="Je scores zijn aantallen mensen, geen delen. Deel elk door het aantal mensen in de helft, `sum(people)`."),
            Mistake("yes-only", wrong=_S_YES_ONLY,
                    en="You count only the people both call yes. The people both call no agree too: below `t`, add the people the forest says no to, `people[e] - forest_yes[e]`.",
                    nl="Je telt alleen de mensen tegen wie allebei ja zeggen. De mensen tegen wie allebei nee zeggen zijn het ook eens: onder `t` tel je de mensen op tegen wie het forest nee zegt, `people[e] - forest_yes[e]`."),
            Mistake("no-only", wrong=_S_NO_ONLY,
                    en="You count only the people both call no. The people both call yes agree too: from `t` on, add the people the forest says yes to, `forest_yes[e]`.",
                    nl="Je telt alleen de mensen tegen wie allebei nee zeggen. De mensen tegen wie allebei ja zeggen zijn het ook eens: vanaf `t` tel je de mensen op tegen wie het forest ja zegt, `forest_yes[e]`."),
            Mistake("running", wrong=_S_RUNNING,
                    en="Your count keeps growing from one `t` to the next, so each score also counts the ones before it. Start `same = 0` inside the loop over `t`, before the loop over `e`.",
                    nl="Je telling blijft groeien van de ene `t` naar de volgende, dus elke score telt ook de vorige mee. Begin `same = 0` binnen de loop over `t`, vóór de loop over `e`."),
            Mistake("inside", wrong=_S_INSIDE,
                    en="Your `scores` has 272 numbers instead of 17: you add to it once for every level, inside the loop over `e`. Add once for each `t`, after the loop over `e`.",
                    nl="Je `scores` heeft 272 getallen in plaats van 17: je voegt er bij elk niveau iets aan toe, binnen de loop over `e`. Voeg één keer per `t` toe, na de loop over `e`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: start a count at 0, and for each level `e` from 0 to 15 add the people for whom the copy says what the forest says: the forest's yeses from `t` on, its noes below `t`. Then add the count divided by `sum(people)` to `scores`.",
                      "nl": "Vervang de `...`: begin een telling bij 0, en tel voor elk niveau `e` van 0 tot 15 de mensen op voor wie de kopie zegt wat het forest zegt: de ja's van het forest vanaf `t`, zijn nee's onder `t`. Voeg daarna de telling gedeeld door `sum(people)` toe aan `scores`."},
            "cleared": _CLEARED,
            "no_value": {"en": "The last line of the box shows nothing. Put `scores` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `scores` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Work out one score by hand: at `t = 1` the copy says yes to everyone, so it agrees with the forest for each person the forest says yes to. Which step is different in your code?",
                      "nl": "Je laatste regel geeft `{got}`. Reken één score met de hand uit: bij `t = 1` zegt de kopie tegen iedereen ja, dus hij is het eens met het forest bij elke persoon tegen wie het forest ja zegt. Welke stap is anders in je code?"},
            "correct": {"en": "Right. The best question is education_num ≥ 14: it agrees with the forest on 0.895 of the training half, and on 0.896 of the test half, where it was not chosen. Saying no to everyone (t = 17) gets 0.853. scikit-learn's tree of depth 1 asks about 12.5 instead, because it chooses by Gini impurity, not by fidelity: 0.881 on the test half.",
                        "nl": "Goed. De beste vraag is education_num ≥ 14: die is het met het forest eens bij 0,895 van de helft om op te trainen, en bij 0,896 van de testhelft, waarop hij niet gekozen is. Tegen iedereen nee zeggen (t = 17) haalt 0,853. De boom met diepte 1 van scikit-learn vraagt in plaats daarvan naar 12,5, omdat hij kiest op Gini impurity, niet op fidelity: 0,881 op de testhelft."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fidelity", _F_START, "empty", None),
    ("fidelity", _F_SOLUTION, "correct", None),
    ("fidelity", _F_HEAD + "    same = 0\n    for a, b in zip(copy, original):\n        if a == b:\n"
                 "            same += 1\n    return same / len(original)\n", "correct", None),
    ("fidelity", _F_COUNT, "wrong", "count"),
    ("fidelity", _F_DIFFER, "wrong", "differ"),
    ("fidelity", _F_YES, "wrong", "yes-only"),
    ("fidelity", _F_WHOLE, "wrong", "whole"),
    ("fidelity", _F_FIRST, "wrong", "first"),
    ("fidelity", _F_HEAD + "    return 0.75\n", "almost", None),
    ("fidelity", _f(ret="    return same / 4\n"), "almost", None),
    ("fidelity", _f(ret="    return\n"), "wrong", "no-return"),
    ("fidelity", _F_HEAD + "    ...\n    return 0.75\n", "wrong", "dots-left"),
    ("stump", _S_START, "empty", None),
    ("stump", _S_SOLUTION, "correct", None),
    ("stump", _S_HEAD + "    same = 0\n    for e in range(16):\n        if e + 1 >= t:\n            same += forest_yes[e]\n"
              "        if e + 1 < t:\n            same += people[e] - forest_yes[e]\n    scores.append(same / 24421)\n"
              "scores\n", "correct", None),
    ("stump", _S_LABELS, "wrong", "labels"),
    ("stump", _S_FLIPPED, "wrong", "flipped"),
    ("stump", _S_SHIFTED, "wrong", "shifted"),
    ("stump", _S_COUNT, "wrong", "count"),
    ("stump", _S_YES_ONLY, "wrong", "yes-only"),
    ("stump", _S_NO_ONLY, "wrong", "no-only"),
    ("stump", _S_RUNNING, "wrong", "running"),
    ("stump", _S_INSIDE, "wrong", "inside"),
    ("stump", _s(add="    scores.append(same / 16)\n"), "wrong", None),
    ("stump", _S_SOLUTION.replace("    same = 0\n", "    same = 0\n    ...\n"), "wrong", "dots-left"),
]
