# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-additive-models: a generalized additive model on three columns of UCI Adult, with one
curve per column, learned the way an Explainable Boosting Machine learns them, by boosting small trees on one column
at a time. The learner adds up the curves to a chance, writes one round of that boosting, reads the curve of age, and
says what a model learns from patients who got better care (after the asthma story of Caruana and colleagues); with the
curves after each number of rounds, what the learner's chance() does to one applicant and to the whole table, and what
each tree of their round adds."""
import functools
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
RATE = 0.5                    # the learning rate of the worked example
ROUNDS = 100                  # its rounds; each round has one tree per column
STOPS = [1, 2, 5, 10, 20, 50, 100]   # the stops of the slider: how many rounds
PERSON = (30, 9, 40)          # the applicant of rai-reason-codes
SPANS = [(17, 90), (1, 16), (1, 99)]   # the values each column takes in the table
LABELS = {"en": ["age", "education_num", "hours_per_week"], "nl": ["age", "education_num", "hours_per_week"]}


@functools.lru_cache(maxsize=1)
def table():
    """The three columns of UCI Adult as an array, and who earns more than 50,000 dollars. Read-only."""
    import pandas as pd
    from everything_ai import paths
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    X = people[COLUMNS].to_numpy()
    y = (people["income"] == ">50K").to_numpy()
    for a in (X, y):
        a.flags.writeable = False
    return X, y


@functools.lru_cache(maxsize=1)
def model() -> dict:
    """The model of the worked example: every score starts at the log-odds of the share who earn more; then 100 rounds,
    each with, for every column in turn, a regression tree of depth 2 on that column alone that learns y - p, of which
    half is added to the score. The start, how often the model is right after each round, how many it says yes to,
    and the curves (one array per column, indexed by the value itself) after each number of rounds in STOPS. About
    a second and a half."""
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    X, y = table()
    start = float(np.log(y.mean() / (1 - y.mean())))
    score = np.zeros(len(y)) + start
    grids = [np.arange(hi + 1).reshape(-1, 1) for lo, hi in SPANS]
    curves = [np.zeros(len(g)) for g in grids]
    out = {"start": start, "right": [], "curves": {}}
    for r in range(ROUNDS):
        for j in range(len(COLUMNS)):
            column = X[:, [j]]
            p = 1 / (1 + np.exp(-score))
            tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(column, y - p)
            score = score + RATE * tree.predict(column)
            curves[j] = curves[j] + RATE * tree.predict(grids[j])
        out["right"].append(float(np.mean((score >= 0) == y)))
        if r + 1 in STOPS:
            kept = [c.copy() for c in curves]
            for c in kept:
                c.flags.writeable = False
            out["curves"][r + 1] = kept
    out["yes"] = int((score >= 0).sum())
    return out


def chance_of(person, rounds: int = ROUNDS) -> float:
    """What the model says for one person: the start plus each column's curve at their value, through the sigmoid."""
    m = model()
    score = m["start"] + sum(float(c[v]) for c, v in zip(m["curves"][rounds], person))
    return 1 / (1 + math.exp(-score))


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _pct(share: float, lang: str) -> str:
    """A share as a percentage with one decimal, also when that decimal is 0: 80.0."""
    text = f"{share * 100:.1f}"
    return text.replace(".", ",") if lang == "nl" else text


def _count(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


# --- the picture of *The idea*: the curves, round after round -----------------------------------------------------

_ROUNDS_WORDS = {
    "en": {"score": "adds to the score", "rounds": "{n} round", "rounds_many": "{n} rounds",
           "alt": "Three charts side by side, one curve each: what age, education_num and hours_per_week add to the "
                  "score after {r}. {shape}",
           "flat": "The curves are still almost flat.",
           "shape": "The age curve rises from 25 to its highest point at {peak} and falls after that; the education "
                    "curve climbs in steps, most at 13 and above; the hours curve rises up to about 50 hours and "
                    "falls a little after that.",
           "said": "After {r}: {trees} small trees, and the model is right for <span class=\"c-param ea-num\">{p}%</span> "
                   "of the people. Always saying no is right for 76.1%."},
    "nl": {"score": "telt op bij de score", "rounds": "{n} ronde", "rounds_many": "{n} rondes",
           "alt": "Drie grafieken naast elkaar, met elk één curve: wat age, education_num en hours_per_week optellen "
                  "bij de score na {r}. {shape}",
           "flat": "De curves zijn nog bijna vlak.",
           "shape": "De curve van leeftijd stijgt van 25 tot haar hoogste punt bij {peak} en daalt daarna; de curve van "
                    "opleiding klimt in treden, het meest vanaf 13; de curve van de uren stijgt tot ongeveer 50 uur en "
                    "daalt daarna een beetje.",
           "said": "Na {r}: {trees} kleine bomen, en het model zit bij <span class=\"c-param ea-num\">{p}%</span> van "
                   "de mensen goed. Altijd nee zeggen zit bij 76,1% goed."},
}


def _curve_panel(j: int, curve, lang: str, y: tuple[float, float], role: str = "param") -> chart.Panel:
    lo, hi = SPANS[j]
    xs = list(range(lo, hi + 1))
    ticks = {0: [20, 40, 60, 80], 1: [1, 4, 8, 12, 16], 2: [1, 20, 40, 60, 80, 99]}[j]
    p = chart.Panel(x=(lo - 0.5, hi + 0.5), y=y, title=LABELS[lang][j], xlabel="", ylabel="", xticks=ticks)
    p.line([lo - 0.5, hi + 0.5], [0, 0], role="axis", width=1)
    if j == 1:   # education takes whole levels only: a step per level
        sx, sy = [], []
        for v in xs:
            sx += [v - 0.5, v + 0.5]
            sy += [float(curve[v])] * 2
        p.line(sx, sy, role=role)
    else:
        p.line(xs, [float(curve[v]) for v in xs], role=role)
    return p


def rounds_picture(n: int, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: the three curves after n rounds, on one scale, and the sentence under them."""
    w = _ROUNDS_WORDS[lang]
    n = int(n)
    m = model()
    r = (w["rounds"] if n == 1 else w["rounds_many"]).format(n=n)
    panels = [_curve_panel(j, m["curves"][n][j], lang, (-2.5, 2)) for j in range(len(COLUMNS))]
    panels[0].ylabel = w["score"]
    peak = max(range(17, 91), key=lambda a: float(m["curves"][n][0][a]))
    alt = w["alt"].format(r=r, shape=w["flat"] if n < 5 else w["shape"].format(peak=peak))
    svg = chart.figure(panels, alt=alt, width=230, height=210)
    said = w["said"].format(r=r, trees=n * len(COLUMNS), p=_pct(m["right"][n - 1], lang))
    return svg, said


# --- exercise 1: adding up the curves ----------------------------------------------------------------------------

_ROWS = {}


def _rows() -> list:
    """The three columns as distinct rows, each with how many people have it and how many of them earn more."""
    if "rows" not in _ROWS:
        import numpy as np
        X, y = table()
        keys, inverse = np.unique(X, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        n = np.bincount(inverse, minlength=len(keys))
        yes = np.bincount(inverse, weights=y.astype(float), minlength=len(keys))
        _ROWS["rows"] = [(tuple(int(v) for v in k), int(c), int(round(t))) for k, c, t in zip(keys, n, yes)]
    return _ROWS["rows"]


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


_CHANCE_WORDS = {
    "en": {"title": "Your chance(), with the curves of the worked example",
           "parts": ["start", "age 30", "education 9", "40 hours", "score"],
           "alt": "A row of bars from left to right: the start, {start}, then what age 30 adds, {a}, education level 9, "
                  "{e}, and 40 hours, {h}, ending at the score {s}. Your function turns it into the chance {c}.",
           "key": "For the applicant aged 30 with education level 9 who works 40 hours, each curve adds its part to "
                  "the score, and your chance() gives {c}. On all 48,842 people of the table, it says yes (a chance "
                  "of 0.5 or more) for {yes} and is right for {right}%."},
    "nl": {"title": "Jouw chance(), met de curves van het uitgewerkte voorbeeld",
           "parts": ["start", "leeftijd 30", "opleiding 9", "40 uur", "score"],
           "alt": "Een rij staven van links naar rechts: de start, {start}, dan wat leeftijd 30 optelt, {a}, "
                  "opleidingsniveau 9, {e}, en 40 uur, {h}, met als eind de score {s}. Je functie maakt er de kans "
                  "{c} van.",
           "key": "Voor de aanvrager van 30 met opleidingsniveau 9 die 40 uur werkt, telt elke curve haar deel op bij "
                  "de score, en je chance() geeft {c}. Bij alle 48.842 mensen van de tabel zegt hij ja (een kans van "
                  "0,5 of meer) bij {yes} en zit hij bij {right}% goed."},
}


def _chance_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    w = _CHANCE_WORDS[seen.lang]
    m = model()
    curves = m["curves"][ROUNDS]
    start = m["start"]

    def run():
        mine = _number(fn(PERSON, start, curves))
        if mine is None or not 0 <= mine <= 1:
            return None
        yes = right = 0
        for row, n, earn in _rows():
            c = _number(fn(row, start, curves))
            if c is None:
                return None
            if c >= 0.5:
                yes, right = yes + n, right + earn
            else:
                right += n - earn
        return mine, yes, right

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    mine, yes, right = got
    parts = [start] + [float(c[v]) for c, v in zip(curves, PERSON)]
    total = sum(parts)
    lows, highs, run_sum = [], [], 0.0
    for i, d in enumerate(parts):
        a, b = (0.0, d) if i == 0 else (run_sum, run_sum + d)
        lows.append(min(a, b))
        highs.append(max(a, b))
        run_sum += d
    bottom = min(min(lows), total) - 0.4
    p = chart.Panel(x=(0, 5), y=(bottom, 0.5), title="", xlabel="", ylabel="score",
                    xticks=[0.5, 1.5, 2.5, 3.5, 4.5], xtick_labels=w["parts"])
    p.line([0, 5], [0, 0], role="axis", width=1)
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        p.rect(i + 0.2, lo, i + 0.8, hi, role="param", solid=True)
        p.text(i + 0.5, lo - 0.18, _show(parts[i], seen.lang, 2), "ink")
    p.rect(4.2, min(total, 0), 4.8, max(total, 0), role="pred", solid=True)
    p.text(4.5, min(total, 0) - 0.18, _show(total, seen.lang, 2), "ink")
    fill = {"start": _show(start, seen.lang), "a": _show(parts[1], seen.lang), "e": _show(parts[2], seen.lang),
            "h": _show(parts[3], seen.lang), "s": _show(total, seen.lang), "c": _show(mine, seen.lang),
            "yes": _count(yes, seen.lang), "right": _pct(right / sum(n for r, n, e in _rows()), seen.lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=520, height=260)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# --- exercise 2: one round ---------------------------------------------------------------------------------------

_ROUND_WORDS = {
    "en": {"title": "Your round: what each tree adds", "score": "adds to the score",
           "alt": "Three charts side by side: what your tree for age, for education_num and for hours_per_week adds to "
                  "the score at each value of its column. Together they move a score by at most {most}.",
           "key": "Each of your trees learned from one column only, so what it adds belongs to that column's curve. "
                  "After one round a score has moved by at most {most}; after 100 rounds, the curves of the worked "
                  "example reach from {low} to {high}."},
    "nl": {"title": "Jouw ronde: wat elke boom optelt", "score": "telt op bij de score",
           "alt": "Drie grafieken naast elkaar: wat je boom voor age, voor education_num en voor hours_per_week "
                  "optelt bij de score, bij elke waarde van zijn kolom. Samen verschuiven ze een score hoogstens {most}.",
           "key": "Elk van je bomen leerde van maar één kolom, dus wat hij optelt hoort bij de curve van die kolom. "
                  "Na één ronde is een score hoogstens {most} verschoven; na 100 rondes lopen de curves van het "
                  "uitgewerkte voorbeeld van {low} tot {high}."},
}


def _round_picture(seen) -> str:
    mine = seen.mine
    if mine is None:
        return ""
    trees, rate = mine.names.get("trees"), mine.names.get("rate", RATE)
    if not isinstance(trees, list) or len(trees) != len(COLUMNS) or _number(rate) is None:
        return ""
    import numpy as np
    steps = []
    for j, item in enumerate(trees):
        if not (isinstance(item, tuple) and len(item) == 2 and item[0] == COLUMNS[j] and hasattr(item[1], "predict")):
            return ""
        lo, hi = SPANS[j]
        try:
            step = float(rate) * np.asarray(item[1].predict(np.arange(hi + 1).reshape(-1, 1)), dtype=float)
        except Exception:   # a tree that learned from another number of columns
            return ""
        if step.shape != (hi + 1,) or not np.all(np.isfinite(step)):
            return ""
        steps.append(step)
    w = _ROUND_WORDS[seen.lang]
    top = max(max(abs(float(s[lo:hi + 1].min())), abs(float(s[lo:hi + 1].max()))) for s, (lo, hi) in zip(steps, SPANS))
    top = max(top, 0.01) * 1.2
    panels = [_curve_panel(j, steps[j], seen.lang, (-top, top)) for j in range(len(COLUMNS))]
    panels[0].ylabel = w["score"]
    most = sum(max(abs(float(s[lo:hi + 1].min())), abs(float(s[lo:hi + 1].max()))) for s, (lo, hi) in zip(steps, SPANS))
    final = model()["curves"][ROUNDS]
    low = min(float(c[lo:hi + 1].min()) for c, (lo, hi) in zip(final, SPANS))
    high = max(float(c[lo:hi + 1].max()) for c, (lo, hi) in zip(final, SPANS))
    fill = {"most": _show(most, seen.lang, 2), "low": _show(low, seen.lang, 2), "high": _show(high, seen.lang, 2)}
    svg = chart.figure(panels, alt=w["alt"].format(**fill), width=230, height=210)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# The lesson is done when these are correct. "read" and "treated" are choice questions.
REQUIRED = ["chance", "round", "read", "treated"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1: adding up the curves
_C_HEAD = ("import numpy as np\n\n\n"
           "def chance(person, start, curves):\n"
           "    # person: one value per feature, such as (30, 9, 40)\n"
           "    # curves: one curve per feature; curves[0][30] is what age 30 adds to the score\n")
_C_START = _C_HEAD + "    ...\n"
_C_BODY = ("    score = start\n"
           "    for i in range(len(person)):\n"
           "        score = score + curves[i][person[i]]\n"
           "    return 1 / (1 + np.exp(-score))\n")
_C_SOLUTION = _C_HEAD + _C_BODY


def _c(body: str) -> str:
    return _C_HEAD + body


# Small made-up curves: the first is the example in the lesson's text.
_C_VISIBLE = [((2, 1), -1.0, [[0.0, 0.5, 1.5], [-0.5, 0.5]])]
_C_HIDDEN = [((0, 0), 0.0, [[0.0, 1.0], [0.0, 2.0]]),
             ((1, 0), 0.0, [[0.0, 1.0], [0.0, 2.0]]),
             ((0, 1), 0.5, [[0.0, 1.0], [0.0, 2.0]]),
             ((1, 2, 0), -2.0, [[0.3, -0.3], [0.0, 0.1, 0.9], [1.0, 2.0]]),
             ((3,), 1.0, [[0.0, 0.0, 0.0, -3.0]]),
             ((2, 2), -1.157, [[-1.0, 0.0, 0.5], [-0.5, 0.0, 0.25]]),
             ((4, 0, 1), 0.2, [[0.0, 0.1, 0.2, 0.3, 0.4], [-1.0], [0.0, -0.6]])]

# --- exercise 2: one round, one column at a time
_RATE_LINE = "rate = 0.5"
_R_HEAD = ("import numpy as np\n"
           "import pandas as pd\n"
           "from sklearn.tree import DecisionTreeRegressor\n\n"
           'people = pd.read_csv("data/adult.csv.gz")\n'
           'names = ["age", "education_num", "hours_per_week"]\n'
           'y = (people["income"] == ">50K").to_numpy()\n'
           "start = np.log(y.mean() / (1 - y.mean()))\n"
           f"{_RATE_LINE}\n"
           "score = np.zeros(len(y)) + start\n"
           "trees = []\n")
_LOOP = "for name in names:\n    column = people[[name]].to_numpy()\n"
_R_START = _R_HEAD + _LOOP + "    p = ...\n    tree = ...\n    score = ...\n    trees.append((name, tree))\nscore\n"


def _r(p: str = "1 / (1 + np.exp(-score))", target: str = "y - p", fit_on: str = "column",
       update: str = "score + rate * tree.predict(column)", outside: bool = False) -> str:
    """The round with one part changed; with `outside`, the chance is worked out once, before the loop."""
    tree = f"DecisionTreeRegressor(max_depth=2, random_state=0).fit({fit_on}, {target})"
    if outside:
        return (_R_HEAD + f"p = {p}\n" + _LOOP + f"    tree = {tree}\n    score = {update}\n"
                "    trees.append((name, tree))\nscore\n")
    return (_R_HEAD + _LOOP + f"    p = {p}\n    tree = {tree}\n    score = {update}\n"
            "    trees.append((name, tree))\nscore\n")


_R_SOLUTION = _r()


def _unchanged(got, text) -> bool:
    """Every score still the start: nothing was added."""
    import numpy as np
    try:
        a = np.asarray(got, dtype=float)
    except Exception:
        return False
    return a.ndim == 1 and len(a) > 1 and bool(np.allclose(a, a[0]))


EXERCISES = {
    "chance": FunctionBox(
        name="chance",
        picture=_chance_picture,
        function="chance",
        start=_C_START,
        solution=_C_SOLUTION,
        visible=_C_VISIBLE,
        hidden=_C_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("no-sigmoid", wrong=_c("    score = start\n    for i in range(len(person)):\n"
                                           "        score = score + curves[i][person[i]]\n    return score\n"),
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the score itself, which can be any number. A chance lies between 0 and 1: the sigmoid turns the score into one, `1 / (1 + np.exp(-score))`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft de score zelf, en die kan elk getal zijn. Een kans ligt tussen 0 en 1: de sigmoid maakt er een van, `1 / (1 + np.exp(-score))`."),
            Mistake("odds", wrong=_c("    score = start\n    for i in range(len(person)):\n"
                                     "        score = score + curves[i][person[i]]\n    return np.exp(score)\n"),
                    en="`{call}` gives {got}, and it should give {want}. Your function gives the odds, $e$ to the power of the score. The chance is the sigmoid of the score: `1 / (1 + np.exp(-score))`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft de odds, $e$ tot de macht van de score. De kans is de sigmoid van de score: `1 / (1 + np.exp(-score))`."),
            Mistake("no-start", wrong=_c("    score = 0\n    for i in range(len(person)):\n"
                                         "        score = score + curves[i][person[i]]\n"
                                         "    return 1 / (1 + np.exp(-score))\n"),
                    en="`{call}` gives {got}, and it should give {want}. Your function leaves out `start`: the score everyone begins with, before any curve adds its part.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie laat `start` weg: de score waarmee iedereen begint, voordat een curve haar deel optelt."),
            Mistake("sign", wrong=_c("    score = start\n    for i in range(len(person)):\n"
                                     "        score = score + curves[i][person[i]]\n"
                                     "    return 1 / (1 + np.exp(score))\n"),
                    en="`{call}` gives {got}, and it should give {want}: 1 minus the chance, so a high score gives a low chance. The sigmoid has a minus sign: `np.exp(-score)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: 1 min de kans, dus een hoge score geeft een lage kans. In de sigmoid staat een minteken: `np.exp(-score)`."),
            Mistake("first-only", wrong=_c("    score = start\n    for i in range(len(person)):\n"
                                           "        score = score + curves[i][person[i]]\n"
                                           "        return 1 / (1 + np.exp(-score))\n"),
                    en="`{call}` gives {got}, and it should give {want}: what you get from the first curve alone. If your `return` stands inside the `for` loop, Python leaves the function after the first feature.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: wat je krijgt van alleen de eerste curve. Als je `return` binnen de `for`-loop staat, verlaat Python de functie na de eerste feature."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the body of `chance`: start at `start`, add for each feature its curve at the person's value, and turn the sum into a chance.",
                      "nl": "Vervang de `...` door de inhoud van `chance`: begin bij `start`, tel voor elke feature haar curve bij de waarde van de persoon op, en maak van de som een kans."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work it out by hand: the start, plus what each curve gives at the person's value, then the sigmoid. Which step is different in your function?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het met de hand uit: de start, plus wat elke curve geeft bij de waarde van de persoon, dan de sigmoid. Welke stap is anders in je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of features, and does it use `start` and `curves` as they are given?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal features, en gebruikt hij `start` en `curves` zoals ze worden meegegeven?"},
            "correct": {"en": "Your `chance` is right in all {n} cases. That is all it takes to predict with an additive model: one lookup per feature, a sum, and the sigmoid. The picture runs it with the curves of the worked example: each curve's part is a number you can point at.",
                        "nl": "Je `chance` klopt in alle {n} gevallen. Meer is er niet nodig om met een additief model te voorspellen: één keer opzoeken per feature, een som, en de sigmoid. De tekening voert hem uit met de curves van het uitgewerkte voorbeeld: het deel van elke curve is een getal dat je kunt aanwijzen."},
        },
    ),
    "round": CodeBox(
        name="round",
        picture=_round_picture,
        start=_R_START,
        solution=_R_SOLUTION,
        keep=[_RATE_LINE],
        # The check also runs the box with another learning rate: a 0.5 typed in by hand does not change with it.
        others=[{"rate": 0.25}],
        first=[_DOTS],
        mistakes=[
            Mistake("columns", error=lambda e: isinstance(e, ValueError) and "features" in str(e) and "expecting" in str(e),
                    en="Your tree learns from one set of columns and predicts from another. In an additive model, each tree learns from its own `column` only, and predicts from that same `column`.",
                    nl="Je boom leert van de ene set kolommen en voorspelt met een andere. In een additief model leert elke boom alleen van zijn eigen `column`, en voorspelt hij met diezelfde `column`."),
            Mistake("stale", wrong=_r(outside=True),
                    en="All three of your trees learned the same residuals: the ones from before the round. Here each tree learns what is left after the trees before it, so work out `p` inside the loop, from the `score` as it is at that moment.",
                    nl="Je drie bomen leerden allemaal dezelfde residuen: die van voor de ronde. Hier leert elke boom wat er over is na de bomen ervoor, dus reken `p` uit binnen de loop, uit de `score` zoals die op dat moment is."),
            Mistake("score-residual", wrong=_r(target="y - score"),
                    en="Your trees learned `y - score`. But the score is on the log-odds, and `y` is 0 or 1: the residual is `y` minus the chance, `y - p`.",
                    nl="Je bomen leerden `y - score`. Maar de score staat op de log-odds, en `y` is 0 of 1: het residu is `y` min de kans, `y - p`."),
            Mistake("labels", wrong=_r(target="y"),
                    en="Your trees learn the labels `y` themselves. In boosting each tree learns what is still wrong: `y - p`.",
                    nl="Je bomen leren de labels `y` zelf. Bij boosting leert elke boom wat nog fout is: `y - p`."),
            Mistake("sign", wrong=_r(target="p - y"),
                    en="Your scores moved away from the labels: people who earn more got a lower score. A residual is the label minus the chance, `y - p`: positive where the chance is too low.",
                    nl="Je scores schoven van de labels weg: mensen die meer verdienen, kregen een lagere score. Een residu is het label min de kans, `y - p`: positief waar de kans te laag is."),
            Mistake("sigmoid", wrong=_r(p="1 / (1 + np.exp(score))"),
                    en="Your `p` is 1 minus the chance: a high score gives a low `p`. The sigmoid has a minus sign: `np.exp(-score)`.",
                    nl="Je `p` is 1 min de kans: een hoge score geeft een lage `p`. In de sigmoid staat een minteken: `np.exp(-score)`."),
            Mistake("odds", wrong=_r(p="np.exp(score)"),
                    en="Your `p` is the odds, $e$ to the power of the score, not the chance. The chance is the sigmoid of the score: `1 / (1 + np.exp(-score))`.",
                    nl="Je `p` is de odds, $e$ tot de macht van de score, niet de kans. De kans is de sigmoid van de score: `1 / (1 + np.exp(-score))`."),
            Mistake("no-rate", wrong=_r(update="score + tree.predict(column)"),
                    en="Your code adds all of what each tree predicts. Boosting adds only a part of it, the learning rate: `rate * tree.predict(column)`.",
                    nl="Je code telt alles op wat elke boom voorspelt. Boosting telt maar een deel ervan op, de learning rate: `rate * tree.predict(column)`."),
            Mistake("all-columns", wrong=_r(fit_on='people[names].to_numpy()',
                                            update='score + rate * tree.predict(people[names].to_numpy())'),
                    en="Your trees learned from all three columns at once. One tree can then ask about age and education together, and what it adds belongs to no single curve. In an additive model each tree sees one column: `column`.",
                    nl="Je bomen leerden van alle drie de kolommen tegelijk. Eén boom kan dan naar leeftijd en opleiding samen vragen, en wat hij optelt hoort bij geen enkele curve. In een additief model ziet elke boom één kolom: `column`."),
            Mistake("replace", wrong=_r(update="rate * tree.predict(column)"),
                    en="Your `score` is only what the last tree adds: close to 0, instead of around the start, −1.157. Add it to the score so far: `score + rate * tree.predict(column)`.",
                    nl="Je `score` is alleen wat de laatste boom optelt: dicht bij 0, in plaats van rond de start, −1,157. Tel het op bij de score tot nu toe: `score + rate * tree.predict(column)`."),
            Mistake("unchanged", matches=_unchanged,
                    en="Every score in your `score` is the same number: nothing that tells people apart was added. Add a part of what each tree predicts: `score + rate * tree.predict(column)`.",
                    nl="Elke score in je `score` is hetzelfde getal: er is niets bij gekomen dat mensen van elkaar onderscheidt. Tel een deel op van wat elke boom voorspelt: `score + rate * tree.predict(column)`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the three `...`: the chance `p` from the score, a tree of depth 2 that learns `y - p` from this one column, and the new score.",
                      "nl": "Vervang de drie `...`: de kans `p` uit de score, een boom met diepte 2 die `y - p` leert van deze ene kolom, en de nieuwe score."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another learning rate by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere learning rate."},
            "no_value": {"en": "The last line of the box shows nothing. Put `score` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `score` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. After one round, `score` is the start plus, for each column in turn, `rate` times what a tree of depth 2 on that column predicts, a tree that learned `y - p` with `p` from the score at that moment. Which step is different in your code?",
                      "nl": "Je laatste regel geeft `{got}`. Na één ronde is `score` de start plus, voor elke kolom om de beurt, `rate` keer wat een boom met diepte 2 op die kolom voorspelt, een boom die `y - p` leerde met `p` uit de score van dat moment. Welke stap is anders in je code?"},
            "other": {"en": "Right here. But with {where}, your `score` is different from what it should be. Do your lines use `rate`, instead of 0.5?",
                      "nl": "Hier goed. Maar met {where} is je `score` anders dan hij moet zijn. Gebruiken je regels `rate`, in plaats van 0,5?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, your code stops with an error. Do you use `rate` in your line, instead of 0.5?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt je code met een fout. Gebruik je `rate` in je regel, in plaats van 0,5?"},
            "correct": {"en": "One round, one column at a time: three small trees, each learning what the trees before it left over. The picture shows what each one adds. 100 rounds like this give the curves of the worked example. The check also tried another learning rate.",
                        "nl": "Eén ronde, één kolom tegelijk: drie kleine bomen, die elk leren wat de bomen ervoor overlieten. De tekening laat zien wat elke boom optelt. 100 rondes zoals deze geven de curves van het uitgewerkte voorbeeld. De check probeerde ook een andere learning rate."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("chance", _C_START, "empty", None),
    ("chance", _C_SOLUTION, "correct", None),
    ("chance", _c("    total = start\n    for i in range(len(curves)):\n        total += curves[i][person[i]]\n"
                  "    odds = np.exp(total)\n    return odds / (1 + odds)\n"), "correct", None),
    ("chance", _c("    score = start\n    for i in range(len(person)):\n        score = score + curves[i][person[i]]\n"
                  "    return score\n"), "wrong", "no-sigmoid"),
    ("chance", _c("    score = start\n    for i in range(len(person)):\n        score = score + curves[i][person[i]]\n"
                  "    return np.exp(score)\n"), "wrong", "odds"),
    ("chance", _c("    score = 0\n    for i in range(len(person)):\n        score = score + curves[i][person[i]]\n"
                  "    return 1 / (1 + np.exp(-score))\n"), "wrong", "no-start"),
    ("chance", _c("    score = start\n    for i in range(len(person)):\n        score = score + curves[i][person[i]]\n"
                  "    return 1 / (1 + np.exp(score))\n"), "wrong", "sign"),
    ("chance", _c("    score = start\n    for i in range(len(person)):\n        score = score + curves[i][person[i]]\n"
                  "        return 1 / (1 + np.exp(-score))\n"), "wrong", "first-only"),
    ("chance", _c("    score = start + curves[0][person[0]] + curves[1][person[1]]\n"
                  "    return 1 / (1 + np.exp(-score))\n"), "almost", None),
    ("chance", _c("    score = start\n    for i in range(len(person)):\n        score = score + curves[i][i]\n"
                  "    return 1 / (1 + np.exp(-score))\n"), "wrong", None),
    ("chance", _c("    score = start\n    for i in range(len(person)):\n        score = score + curves[i][person[i]]\n"),
     "wrong", "no-return"),
    ("chance", _C_HEAD + "    ...\n    return 0.5\n", "wrong", "dots-left"),
    ("round", _R_START, "empty", None),
    ("round", _R_SOLUTION, "correct", None),
    ("round", _R_HEAD + _LOOP + "    p = 1 / (1 + np.exp(-score))\n    residuals = y - p\n"
              "    tree = DecisionTreeRegressor(max_depth=2, random_state=0)\n    tree.fit(column, residuals)\n"
              "    score += rate * tree.predict(column)\n    trees.append((name, tree))\nscore\n", "correct", None),
    ("round", _r(outside=True), "wrong", "stale"),
    ("round", _r(target="y - score"), "wrong", "score-residual"),
    ("round", _r(target="y"), "wrong", "labels"),
    ("round", _r(target="p - y"), "wrong", "sign"),
    ("round", _r(p="1 / (1 + np.exp(score))"), "wrong", "sigmoid"),
    ("round", _r(p="np.exp(score)"), "wrong", "odds"),
    ("round", _r(update="score + tree.predict(column)"), "wrong", "no-rate"),
    ("round", _r(fit_on='people[names].to_numpy()',
                 update='score + rate * tree.predict(people[names].to_numpy())'), "wrong", "all-columns"),
    ("round", _r(fit_on='people[names].to_numpy()'), "error", "columns"),
    ("round", _r(update="rate * tree.predict(column)"), "wrong", "replace"),
    ("round", _r(update="score"), "wrong", "unchanged"),
    ("round", _r(update="score + 0.5 * tree.predict(column)"), "almost", "other-value"),
    ("round", _r().replace(_RATE_LINE, "rate = 0.1"), "wrong", "changed-line"),
    ("round", _R_SOLUTION.replace("    trees.append", "    ...\n    trees.append"), "wrong", "dots-left"),
]
